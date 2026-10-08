"""Foundational components for the educational custom Transformer."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class TokenEmbedding(nn.Module):
    """Map each integer token ID to a learnable dense vector."""

    def __init__(self, vocabulary_size: int, embedding_dimension: int) -> None:
        super().__init__()
        if vocabulary_size <= 0:
            raise ValueError("vocabulary_size must be positive")
        if embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be positive")
        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension)

    def forward(self, token_ids: Tensor) -> Tensor:
        """Return embeddings with shape ``(batch, sequence, embedding_dimension)``."""
        if token_ids.dtype != torch.long:
            raise TypeError("token_ids must have dtype torch.long")
        return self.embedding(token_ids)


class PositionalEncoding(nn.Module):
    """Add fixed sinusoidal position information to token embeddings.

    Self-attention does not inherently know token order. The sine/cosine
    signals give every sequence position a distinct, reusable representation.
    For position ``p`` and dimension ``i``:

    ``PE(p, 2i) = sin(p / 10000 ** (2i / d_model))``
    ``PE(p, 2i+1) = cos(p / 10000 ** (2i / d_model))``
    """

    def __init__(
        self,
        embedding_dimension: int,
        max_sequence_length: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be positive")
        if max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")

        positions = torch.arange(max_sequence_length, dtype=torch.float32).unsqueeze(1)
        dimensions = torch.arange(0, embedding_dimension, 2, dtype=torch.float32)
        frequencies = torch.exp(
            -math.log(10000.0) * dimensions / embedding_dimension
        )
        encoding = torch.zeros(max_sequence_length, embedding_dimension)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, embeddings: Tensor) -> Tensor:
        """Add positional signals to embeddings with shape ``(batch, sequence, dim)``."""
        if embeddings.ndim != 3:
            raise ValueError("embeddings must have shape (batch, sequence, dimension)")
        sequence_length = embeddings.shape[1]
        if sequence_length > self.encoding.shape[1]:
            raise ValueError(
                f"sequence length {sequence_length} exceeds configured maximum "
                f"{self.encoding.shape[1]}"
            )
        return self.dropout(embeddings + self.encoding[:, :sequence_length].to(embeddings))


class SelfAttention(nn.Module):
    """Compute scaled dot-product self-attention without a ready-made module.

    Every token produces:

    * ``Q`` (query): what this token is looking for
    * ``K`` (key): what this token offers for matching
    * ``V`` (value): the information passed to the output

    The attention equation is:
    ``softmax(QKᵀ / sqrt(d_k))V``.
    """

    def __init__(self, embedding_dimension: int) -> None:
        super().__init__()
        if embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be positive")
        self.embedding_dimension = embedding_dimension
        self.query_projection = nn.Linear(embedding_dimension, embedding_dimension)
        self.key_projection = nn.Linear(embedding_dimension, embedding_dimension)
        self.value_projection = nn.Linear(embedding_dimension, embedding_dimension)

    def forward(
        self,
        embeddings: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return attended embeddings and attention weights.

        ``embeddings`` has shape ``(batch, sequence, dimension)``.
        A boolean mask uses ``True`` for positions that may be attended to.
        It may have shape ``(sequence, sequence)`` or broadcastable batch
        dimensions such as ``(batch, sequence, sequence)``.
        """
        if embeddings.ndim != 3:
            raise ValueError("embeddings must have shape (batch, sequence, dimension)")
        if embeddings.shape[-1] != self.embedding_dimension:
            raise ValueError(
                f"expected embedding dimension {self.embedding_dimension}, "
                f"got {embeddings.shape[-1]}"
            )

        queries = self.query_projection(embeddings)
        keys = self.key_projection(embeddings)
        values = self.value_projection(embeddings)

        # Scaling prevents large dot products from making softmax gradients
        # extremely small when the embedding dimension grows.
        scores = torch.matmul(queries, keys.transpose(-2, -1))
        scores = scores / math.sqrt(self.embedding_dimension)

        if attention_mask is not None:
            if attention_mask.dtype != torch.bool:
                raise TypeError("attention_mask must have dtype torch.bool")
            scores = scores.masked_fill(~attention_mask, torch.finfo(scores.dtype).min)

        # Softmax turns scores into weights that sum to one for each query.
        attention_weights = torch.softmax(scores, dim=-1)
        attended_embeddings = torch.matmul(attention_weights, values)
        return attended_embeddings, attention_weights


class MultiHeadAttention(nn.Module):
    """Run scaled dot-product attention across several independent heads."""

    def __init__(self, embedding_dimension: int, number_of_heads: int) -> None:
        super().__init__()
        if embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be positive")
        if number_of_heads <= 0:
            raise ValueError("number_of_heads must be positive")
        if embedding_dimension % number_of_heads != 0:
            raise ValueError("embedding_dimension must be divisible by number_of_heads")

        self.embedding_dimension = embedding_dimension
        self.number_of_heads = number_of_heads
        self.head_dimension = embedding_dimension // number_of_heads
        self.query_projection = nn.Linear(embedding_dimension, embedding_dimension)
        self.key_projection = nn.Linear(embedding_dimension, embedding_dimension)
        self.value_projection = nn.Linear(embedding_dimension, embedding_dimension)
        self.output_projection = nn.Linear(embedding_dimension, embedding_dimension)

    def _split_heads(self, values: Tensor) -> Tensor:
        batch_size, sequence_length, _ = values.shape
        values = values.reshape(
            batch_size,
            sequence_length,
            self.number_of_heads,
            self.head_dimension,
        )
        return values.transpose(1, 2)

    def forward(
        self,
        embeddings: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return output and weights shaped ``(batch, heads, sequence, sequence)``."""
        if embeddings.ndim != 3:
            raise ValueError("embeddings must have shape (batch, sequence, dimension)")
        if embeddings.shape[-1] != self.embedding_dimension:
            raise ValueError(
                f"expected embedding dimension {self.embedding_dimension}, "
                f"got {embeddings.shape[-1]}"
            )

        queries = self._split_heads(self.query_projection(embeddings))
        keys = self._split_heads(self.key_projection(embeddings))
        values = self._split_heads(self.value_projection(embeddings))

        # Each head computes attention independently in a smaller subspace.
        scores = torch.matmul(queries, keys.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dimension)
        if attention_mask is not None:
            if attention_mask.dtype != torch.bool:
                raise TypeError("attention_mask must have dtype torch.bool")
            scores = scores.masked_fill(~attention_mask, torch.finfo(scores.dtype).min)

        attention_weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(attention_weights, values)
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.reshape(
            embeddings.shape[0], embeddings.shape[1], self.embedding_dimension
        )
        return self.output_projection(attended), attention_weights


class FeedForwardNetwork(nn.Module):
    """Apply the position-wise two-layer feed-forward transformation."""

    def __init__(
        self,
        embedding_dimension: int,
        hidden_dimension: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be positive")
        hidden_dimension = hidden_dimension or embedding_dimension * 4
        if hidden_dimension <= 0:
            raise ValueError("hidden_dimension must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")

        self.network = nn.Sequential(
            nn.Linear(embedding_dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, embedding_dimension),
        )

    def forward(self, embeddings: Tensor) -> Tensor:
        """Transform each sequence position independently."""
        return self.network(embeddings)


class TransformerBlock(nn.Module):
    """One encoder block with attention, residual paths, and normalization.

    The residual connections preserve the input signal while each sublayer
    learns an improvement. LayerNorm stabilizes the activations passed between
    sublayers, which makes optimization more reliable.
    """

    def __init__(
        self,
        embedding_dimension: int,
        number_of_heads: int,
        feed_forward_dimension: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(embedding_dimension, number_of_heads)
        self.attention_norm = nn.LayerNorm(embedding_dimension)
        self.feed_forward = FeedForwardNetwork(
            embedding_dimension,
            feed_forward_dimension,
            dropout,
        )
        self.feed_forward_norm = nn.LayerNorm(embedding_dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        embeddings: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return transformed embeddings and the block's attention weights."""
        attended, attention_weights = self.attention(embeddings, attention_mask)
        embeddings = self.attention_norm(embeddings + self.dropout(attended))

        transformed = self.feed_forward(embeddings)
        embeddings = self.feed_forward_norm(embeddings + self.dropout(transformed))
        return embeddings, attention_weights


class CustomTransformer(nn.Module):
    """A small decoder-style Transformer for next-token prediction."""

    def __init__(
        self,
        vocabulary_size: int,
        embedding_dimension: int = 64,
        number_of_heads: int = 4,
        number_of_layers: int = 2,
        feed_forward_dimension: int | None = None,
        max_sequence_length: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if vocabulary_size <= 0:
            raise ValueError("vocabulary_size must be positive")
        if number_of_layers <= 0:
            raise ValueError("number_of_layers must be positive")

        self.max_sequence_length = max_sequence_length
        self.embedding = TokenEmbedding(vocabulary_size, embedding_dimension)
        self.position = PositionalEncoding(
            embedding_dimension,
            max_sequence_length,
            dropout,
        )
        self.blocks = nn.ModuleList(
            TransformerBlock(
                embedding_dimension,
                number_of_heads,
                feed_forward_dimension,
                dropout,
            )
            for _ in range(number_of_layers)
        )
        self.final_norm = nn.LayerNorm(embedding_dimension)
        self.output_projection = nn.Linear(embedding_dimension, vocabulary_size)

    def _causal_mask(self, sequence_length: int, device: torch.device) -> Tensor:
        """Prevent a token from attending to future tokens during training."""
        return torch.tril(
            torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=device)
        )

    def forward(self, token_ids: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Return next-token logits and attention maps for each Transformer block."""
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, sequence)")
        sequence_length = token_ids.shape[1]
        if sequence_length > self.max_sequence_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds configured maximum "
                f"{self.max_sequence_length}"
            )

        hidden_states = self.position(self.embedding(token_ids))
        mask = self._causal_mask(sequence_length, token_ids.device)
        attention_maps = []
        for block in self.blocks:
            hidden_states, attention_weights = block(hidden_states, mask)
            attention_maps.append(attention_weights)

        logits = self.output_projection(self.final_norm(hidden_states))
        return logits, attention_maps

    @torch.no_grad()
    def generate(
        self,
        token_ids: Tensor,
        number_of_tokens: int,
        end_token_id: int | None = None,
    ) -> Tensor:
        """Generate tokens autoregressively from a starting sequence."""
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, sequence)")
        if number_of_tokens < 0:
            raise ValueError("number_of_tokens must not be negative")

        was_training = self.training
        self.eval()
        generated = token_ids
        for _ in range(number_of_tokens):
            context = generated[:, -self.max_sequence_length :]
            logits, _ = self(context)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=1)
            if end_token_id is not None and torch.all(next_token == end_token_id):
                break
        self.train(was_training)
        return generated


if __name__ == "__main__":
    embedding = TokenEmbedding(vocabulary_size=41, embedding_dimension=16)
    positions = PositionalEncoding(embedding_dimension=16, max_sequence_length=8)
    token_ids = torch.tensor([[2, 29, 30, 31, 12, 3]], dtype=torch.long)
    output = positions(embedding(token_ids))
    assert output.shape == (1, 6, 16)
    assert not torch.equal(output[:, 0], output[:, 1])
    print(f"Embedding output shape: {tuple(output.shape)}")
    print("Token embedding and positional encoding smoke test passed.")

    attention = SelfAttention(embedding_dimension=16)
    attended, weights = attention(output)
    assert attended.shape == output.shape
    assert weights.shape == (1, 6, 6)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1, 6), atol=1e-6)
    print("Scaled dot-product self-attention smoke test passed.")

    multi_head_attention = MultiHeadAttention(embedding_dimension=16, number_of_heads=4)
    multi_head_output, multi_head_weights = multi_head_attention(output)
    assert multi_head_output.shape == output.shape
    assert multi_head_weights.shape == (1, 4, 6, 6)
    assert torch.allclose(
        multi_head_weights.sum(dim=-1), torch.ones(1, 4, 6), atol=1e-6
    )
    print("Multi-head attention smoke test passed.")

    block_input = output.detach().requires_grad_(True)
    transformer_block = TransformerBlock(
        embedding_dimension=16,
        number_of_heads=4,
        feed_forward_dimension=32,
    )
    block_output, block_weights = transformer_block(block_input)
    assert block_output.shape == block_input.shape
    assert block_weights.shape == (1, 4, 6, 6)
    block_output.mean().backward()
    assert block_input.grad is not None
    print("Feed-forward, residual, LayerNorm, and block smoke test passed.")

    transformer = CustomTransformer(
        vocabulary_size=41,
        embedding_dimension=16,
        number_of_heads=4,
        number_of_layers=2,
        feed_forward_dimension=32,
        max_sequence_length=8,
    )
    logits, attention_maps = transformer(token_ids)
    assert logits.shape == (1, 6, 41)
    assert len(attention_maps) == 2
    generated = transformer.generate(token_ids[:, :2], number_of_tokens=3)
    assert generated.shape == (1, 5)
    print("Complete custom Transformer forward and generation smoke test passed.")
