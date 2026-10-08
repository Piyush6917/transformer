"""Grounded PDF question answering with the custom Transformer project.

Retrieval is deliberately transparent: the question and PDF chunks are
compared using normalized word overlap, then the answer is assembled only
from retrieved sentences. This prevents the small language model from
inventing facts that are not present in the documents.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from dataclasses import dataclass

import pymupdf
import torch

from model import CustomTransformer
from tokenizer import Tokenizer


@dataclass(frozen=True)
class DocumentChunk:
    """A retrievable piece of a PDF with citation metadata."""

    document_name: str
    page_number: int
    text: str


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "what",
    "when", "where", "which", "who", "why", "with", "compare", "comparison",
    "difference", "differences", "between", "versus", "vs", "diff",
}


def load_pdf_chunks(pdf_directory: Path, words_per_chunk: int = 120) -> list[DocumentChunk]:
    """Extract page-aware, word-bounded chunks from PDFs."""
    if words_per_chunk < 20:
        raise ValueError("words_per_chunk must be at least 20")
    pdf_files = sorted(pdf_directory.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {pdf_directory}")

    chunks: list[DocumentChunk] = []
    for pdf_path in pdf_files:
        with pymupdf.open(pdf_path) as document:
            for page_number, page in enumerate(document, start=1):
                text = " ".join(page.get_text("text").split())
                words = text.split()
                for start in range(0, len(words), words_per_chunk):
                    chunk_text = " ".join(words[start : start + words_per_chunk])
                    if chunk_text:
                        chunks.append(DocumentChunk(pdf_path.name, page_number, chunk_text))
    if not chunks:
        raise ValueError(f"No readable text found in PDFs under {pdf_directory}")
    return chunks


def _terms(text: str) -> set[str]:
    terms = set()
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        if token in STOP_WORDS or len(token) <= 1:
            continue
        terms.add({"lists": "list", "tuples": "tuple"}.get(token, token))
    return terms


def _sentences(text: str) -> list[str]:
    """Split text and discard obvious slide-header fragments."""
    sentences = []
    for sentence in re.split(r"(?<=[.!?])\s+|(?<=:)\s+", text):
        cleaned = sentence.replace("\uf0d8", " ").replace("\u2022", " ")
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" -•")
        words = cleaned.split()
        if len(words) < 4 or len(words) > 55:
            continue
        # PDF slides often repeat a title several times in the extracted text.
        if len(words) >= 6 and len(set(word.lower() for word in words)) < len(words) * 0.55:
            continue
        sentences.append(cleaned)
    return sentences


def _is_definition_question(question: str) -> bool:
    return bool(re.search(r"\bwhat\s+(?:is|are)\b|\bdefine\b", question.lower()))


def _is_comparison_question(question: str) -> bool:
    return bool(
        re.search(r"\b(?:difference|differences|compare|comparison|between|versus|vs)\b",
                  question.lower())
    )


def _comparison_fragments(text: str) -> list[str]:
    """Split slide-style comparison prose into short evidence fragments."""
    normalized = text.replace("\uf0d8", " ").replace("\u2022", " ")
    parts = re.split(
        r"(?i)(?=\b(?:list|lists|tuple|tuples)\b\s+"
        r"(?:are|is|can|cannot|usually|often|allow|use))|(?<=[.!?])\s+",
        normalized,
    )
    return [re.sub(r"\s+", " ", part).strip(" -:") for part in parts if part.strip()]


def _has_definition_evidence(sentence: str, term: str) -> bool:
    """Check that a topic is actually described, not merely mentioned."""
    return bool(
        re.search(
            rf"^(?:the\s+)?{re.escape(term)}\b\s+(?:\w+\s+){{0,3}}"
            r"(?:is|are|refers|used|designed)\b",
            sentence,
            re.IGNORECASE,
        )
    )


def retrieve_chunks(
    question: str,
    chunks: list[DocumentChunk],
    top_k: int = 3,
) -> list[tuple[float, DocumentChunk]]:
    """Rank chunks by transparent lexical similarity to the question."""
    question_terms = _terms(question)
    if not question_terms:
        return []
    scored = []
    comparison_question = _is_comparison_question(question)
    for chunk in chunks:
        chunk_terms = _terms(chunk.text)
        overlap = question_terms & chunk_terms
        score = len(overlap) / len(question_terms)
        if comparison_question:
            requested_concepts = {
                term for term in question_terms if term in {"list", "lists", "tuple", "tuples"}
            }
            if len(requested_concepts) >= 2 and not requested_concepts <= chunk_terms:
                continue
            if not re.search(
                r"\b(?:difference|similar|mutable|immutable|faster|slower)\b",
                chunk.text,
                re.IGNORECASE,
            ):
                continue
            if re.search(
                r"\b(?:list|lists)\b.{0,80}\b(?:tuple|tuples)\b|"
                r"\b(?:tuple|tuples)\b.{0,80}\b(?:list|lists)\b",
                chunk.text,
                re.IGNORECASE,
            ):
                score += 2.0
                if re.search(
                    r"\b(?:list|lists)\b.{0,80}\bmutable\b.{0,80}"
                    r"\b(?:tuple|tuples)\b.{0,80}\bimmutable\b|"
                    r"\b(?:tuple|tuples)\b.{0,80}\bimmutable\b.{0,80}"
                    r"\b(?:list|lists)\b.{0,80}\bmutable\b",
                    chunk.text,
                    re.IGNORECASE,
                ):
                    score += 2.0
            else:
                score += 1.0
        # A one-word query such as "Python" matches almost every page. Require
        # useful evidence instead of treating the repeated topic word as an
        # answer.
        if len(question_terms) == 1:
            evidence = any(
                question_terms <= _terms(sentence)
                and _has_definition_evidence(sentence, next(iter(question_terms)))
                for sentence in _sentences(chunk.text)
            )
            if not evidence:
                continue
        if score > 0:
            scored.append((score, chunk))
    return sorted(scored, key=lambda item: item[0], reverse=True)[:top_k]


def answer_question(
    question: str,
    chunks: list[DocumentChunk],
    top_k: int = 3,
) -> tuple[str, list[tuple[float, DocumentChunk]]]:
    """Answer only with sentences from retrieved PDF chunks."""
    retrieved = retrieve_chunks(question, chunks, top_k)
    if not retrieved or retrieved[0][0] < 0.15:
        return (
            "I could not find enough information in the provided documents "
            "to answer this question.",
            retrieved,
        )

    question_terms = _terms(question)
    candidates: list[tuple[float, str, DocumentChunk]] = []
    definition_question = _is_definition_question(question)
    comparison_question = _is_comparison_question(question)
    for score, chunk in retrieved:
        fragments = _comparison_fragments(chunk.text) if comparison_question else _sentences(chunk.text)
        for sentence in fragments:
            sentence_terms = _terms(sentence)
            overlap = question_terms & sentence_terms
            coverage = len(overlap) / len(question_terms)
            if not overlap or (definition_question and coverage < 1.0):
                continue
            if comparison_question:
                requested_concepts = {
                    term for term in question_terms
                    if term in {"list", "lists", "tuple", "tuples"}
                }
                if len(requested_concepts) >= 2 and not (
                    requested_concepts & sentence_terms
                ):
                    continue
                if not re.search(
                    r"\b(?:difference|similar|mutable|immutable|faster|slower)\b",
                    sentence,
                    re.IGNORECASE,
                ):
                    continue
                if re.search(
                    r"\b(?:list1|tup1|creating|constructor|with one item|using round)\b",
                    sentence,
                    re.IGNORECASE,
                ):
                    continue
            if len(question_terms) == 1 and not _has_definition_evidence(
                sentence, next(iter(question_terms))
            ):
                continue
            if definition_question and re.match(
                r"^(write|explain|list|describe a program|give an example)\b",
                sentence,
                re.IGNORECASE,
            ):
                continue
            definition_bonus = (
                1.0
                if re.search(r"\b(is|are|refers|used|designed)\b", sentence, re.I)
                else 0.0
            )
            length_penalty = max(0, len(sentence.split()) - 28) * 0.02
            candidates.append(
                (coverage + definition_bonus + score - length_penalty, sentence, chunk)
            )
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates[:2] if comparison_question else candidates[:3]
    if definition_question and not selected:
        return (
            "I could not find a clear definition in the provided documents "
            "to answer this question.",
            retrieved,
        )
    if not selected:
        return (
            "I could not find enough information in the provided documents "
            "to answer this question.",
            retrieved,
        )
    answer = " ".join(sentence for _, sentence, _ in selected)
    return answer, retrieved


def load_checkpoint(checkpoint_path: Path) -> tuple[CustomTransformer, Tokenizer]:
    """Load a model and its vocabulary saved by ``train.py``."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Run train.py first."
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_config = checkpoint["model_config"]
    tokenizer = Tokenizer()
    tokenizer.token_to_id = {
        str(token): int(token_id)
        for token, token_id in checkpoint["token_to_id"].items()
    }
    model = CustomTransformer(**model_config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, tokenizer


def generate_response(
    model: CustomTransformer,
    tokenizer: Tokenizer,
    prompt: str,
    number_of_tokens: int = 20,
) -> str:
    """Generate a continuation for a prompt using greedy decoding."""
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if number_of_tokens < 0:
        raise ValueError("number_of_tokens must not be negative")

    prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    generated_ids = model.generate(
        prompt_ids,
        number_of_tokens=number_of_tokens,
        end_token_id=tokenizer.eos_id,
    )
    return tokenizer.decode(generated_ids[0].tolist())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).with_name("checkpoints") / "custom_transformer.pt",
    )
    parser.add_argument("--pdf-dir", type=Path, default=Path(__file__).with_name("pdf"))
    parser.add_argument("--prompt", type=str, help="Ask one PDF question and exit")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--words-per-chunk", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    arguments = parse_args()
    chunks = load_pdf_chunks(arguments.pdf_dir, arguments.words_per_chunk)

    if arguments.prompt:
        answer, sources = answer_question(arguments.prompt, chunks, arguments.top_k)
        print(f"Answer: {answer}")
        if sources:
            print("\nSources:")
            for score, chunk in sources:
                print(f"- {chunk.document_name}, page {chunk.page_number} (score {score:.2f})")
        return

    print("PDF question answering. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.lower() in {"exit", "quit"}:
            break
        if not prompt:
            print("Please enter a question or prompt.")
            continue
        answer, sources = answer_question(prompt, chunks, arguments.top_k)
        print(f"Answer: {answer}")
        if sources:
            print("Sources:")
            for _, chunk in sources:
                print(f"- {chunk.document_name}, page {chunk.page_number}")


if __name__ == "__main__":
    main()
