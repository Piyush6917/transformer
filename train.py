"""Train the custom Transformer on the local next-token dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import fitz
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from model import CustomTransformer
from tokenizer import Tokenizer


def extract_pdf_text(pdf_directory: Path) -> str:
    """Extract readable text from every PDF page in a directory."""
    pdf_files = sorted(pdf_directory.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {pdf_directory}")

    pages: list[str] = []
    for pdf_path in pdf_files:
        with fitz.open(pdf_path) as document:
            for page_number, page in enumerate(document, start=1):
                page_text = " ".join(page.get_text("text").split())
                if page_text:
                    pages.append(
                        f"[Document: {pdf_path.name} | Page: {page_number}]\n{page_text}"
                    )

    if not pages:
        raise ValueError(f"No readable text found in PDFs under {pdf_directory}")
    return "\n\n".join(pages)


def prepare_pdf_data(pdf_directory: Path, data_path: Path) -> None:
    """Extract PDF text and replace the training corpus at ``data_path``."""
    text = extract_pdf_text(pdf_directory)
    data_path.write_text(text, encoding="utf-8")
    print(f"Extracted {len(text):,} characters to {data_path}")


class NextTokenDataset(Dataset[tuple[Tensor, Tensor]]):
    """Create input/target pairs where the target is shifted one token ahead."""

    def __init__(self, token_ids: list[int], sequence_length: int) -> None:
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        if len(token_ids) < sequence_length:
            raise ValueError("dataset does not contain enough tokens")
        self.inputs = []
        self.targets = []
        for start in range(len(token_ids) - sequence_length + 1):
            window = token_ids[start : start + sequence_length]
            self.inputs.append(torch.tensor(window[:-1], dtype=torch.long))
            self.targets.append(torch.tensor(window[1:], dtype=torch.long))

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.inputs[index], self.targets[index]


def train(
    data_path: Path,
    checkpoint_path: Path,
    *,
    sequence_length: int = 16,
    embedding_dimension: int = 64,
    number_of_heads: int = 4,
    number_of_layers: int = 2,
    epochs: int = 100,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
) -> None:
    """Train and save a small language model."""
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")

    text = data_path.read_text(encoding="utf-8")
    tokenizer = Tokenizer.from_text(text)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    dataset = NextTokenDataset(token_ids, sequence_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = CustomTransformer(
        vocabulary_size=len(tokenizer),
        embedding_dimension=embedding_dimension,
        number_of_heads=number_of_heads,
        number_of_layers=number_of_layers,
        max_sequence_length=sequence_length - 1,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for inputs, targets in loader:
            optimizer.zero_grad()
            logits, _ = model(inputs)
            loss = loss_function(logits.reshape(-1, len(tokenizer)), targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        if epoch == 1 or epoch % max(1, epochs // 10) == 0:
            print(f"Epoch {epoch:3d}/{epochs} - loss: {total_loss / len(loader):.4f}")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "token_to_id": tokenizer.token_to_id,
            "model_config": {
                "vocabulary_size": len(tokenizer),
                "embedding_dimension": embedding_dimension,
                "number_of_heads": number_of_heads,
                "number_of_layers": number_of_layers,
                "max_sequence_length": sequence_length - 1,
            },
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("data.txt"))
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        help="Extract PDF text from this directory before training.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).with_name("checkpoints") / "custom_transformer.pt",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.pdf_dir:
        prepare_pdf_data(arguments.pdf_dir, arguments.data)
    train(
        arguments.data,
        arguments.checkpoint,
        epochs=arguments.epochs,
        sequence_length=arguments.sequence_length,
        batch_size=arguments.batch_size,
    )
