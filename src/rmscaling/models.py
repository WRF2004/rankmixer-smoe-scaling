"""Small, reproducible CTR rankers and a genuinely sparse RankMixer variant.

The RankMixer block follows equations (1), (3)--(9) of
https://arxiv.org/html/2507.15551. Its optional top-k expert layer is an
experimental sparse-train/sparse-infer variant, not the paper's ReLU router.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _groups(num_fields: int, token_groups: object) -> tuple[tuple[int, ...], ...]:
    if token_groups is None:
        return tuple((i,) for i in range(num_fields))
    raw = token_groups.values() if isinstance(token_groups, Mapping) else token_groups
    groups = tuple(tuple(int(i) for i in group) for group in raw)
    flattened = [i for group in groups for i in group]
    if any(not group for group in groups) or sorted(flattened) != list(range(num_fields)):
        raise ValueError("token_groups must partition all categorical fields exactly once")
    return groups


class FeatureTokenizer(nn.Module):
    """Embed each categorical field, then optionally combine semantic groups.

    The numerical fields are projected together into one additional token.
    Field ID zero can be used as a missing/unknown category by the data layer.
    """

    def __init__(
        self,
        vocab_sizes: Sequence[int],
        num_dense: int,
        embedding_dim: int,
        token_groups: object = None,
    ) -> None:
        super().__init__()
        if embedding_dim < 1 or num_dense < 0 or any(int(v) < 1 for v in vocab_sizes):
            raise ValueError("embedding_dim and vocab sizes must be positive; num_dense cannot be negative")
        self.vocab_sizes = tuple(int(v) for v in vocab_sizes)
        self.num_dense = int(num_dense)
        self.embedding_dim = int(embedding_dim)
        self.group_indices = _groups(len(self.vocab_sizes), token_groups)
        self.num_tokens = len(self.group_indices) + int(self.num_dense > 0)
        if not self.num_tokens:
            raise ValueError("at least one categorical or numerical feature is required")

        self.embeddings = nn.ModuleList(
            nn.Embedding(v, self.embedding_dim) for v in self.vocab_sizes
        )
        for table in self.embeddings:
            nn.init.normal_(table.weight, mean=0.0, std=0.01)
        self.group_projections = nn.ModuleList(
            nn.Identity() if len(group) == 1 else nn.Linear(len(group) * self.embedding_dim, self.embedding_dim)
            for group in self.group_indices
        )
        self.numerical_projection = (
            nn.Linear(self.num_dense, self.embedding_dim) if self.num_dense else None
        )

    @property
    def embedding_parameter_count(self) -> int:
        return sum(p.numel() for table in self.embeddings for p in table.parameters())

    def forward(self, batch: Mapping[str, Tensor]) -> Tensor:
        categorical = batch["categorical"]
        numerical = batch["numerical"]
        if categorical.ndim != 2 or categorical.shape[1] != len(self.vocab_sizes):
            raise ValueError("categorical must have shape [batch, number of fields]")
        if numerical.ndim != 2 or numerical.shape != (categorical.shape[0], self.num_dense):
            raise ValueError("numerical must have shape [batch, num_dense]")
        if categorical.shape[0] == 0:
            raise ValueError("empty batches are not supported")

        field_embeddings = [table(categorical[:, i]) for i, table in enumerate(self.embeddings)]
        tokens = [
            projection(torch.cat([field_embeddings[i] for i in group], dim=-1))
            for group, projection in zip(self.group_indices, self.group_projections)
        ]
        if self.numerical_projection is not None:
            tokens.append(self.numerical_projection(numerical))
        return torch.stack(tokens, dim=1)


def _mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> nn.Sequential:
    dims = [input_dim, *(int(h) for h in hidden_dims), output_dim]
    if any(d < 1 for d in dims):
        raise ValueError("all layer dimensions must be positive")
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


class MLPRanker(nn.Module):
    """Embedding + MLP CTR baseline."""

    def __init__(
        self,
        vocab_sizes: Sequence[int],
        num_dense: int,
        embedding_dim: int = 32,
        hidden_dims: Sequence[int] = (256, 128),
        token_groups: object = None,
    ) -> None:
        super().__init__()
        self.tokenizer = FeatureTokenizer(vocab_sizes, num_dense, embedding_dim, token_groups)
        self.network = _mlp(self.tokenizer.num_tokens * embedding_dim, hidden_dims, 1)

    def forward(self, batch: Mapping[str, Tensor]) -> Tensor:
        x = self.tokenizer(batch).flatten(start_dim=1)
        return self.network(x).squeeze(-1)

    @property
    def embedding_parameter_count(self) -> int:
        return self.tokenizer.embedding_parameter_count


class DCNv2Ranker(nn.Module):
    """Full-rank DCNv2-style cross network in parallel with an MLP."""

    def __init__(
        self,
        vocab_sizes: Sequence[int],
        num_dense: int,
        embedding_dim: int = 32,
        cross_depth: int = 2,
        hidden_dims: Sequence[int] = (256, 128),
        token_groups: object = None,
    ) -> None:
        super().__init__()
        if cross_depth < 1:
            raise ValueError("cross_depth must be positive")
        self.tokenizer = FeatureTokenizer(vocab_sizes, num_dense, embedding_dim, token_groups)
        input_dim = self.tokenizer.num_tokens * embedding_dim
        self.cross_layers = nn.ModuleList(nn.Linear(input_dim, input_dim) for _ in range(cross_depth))
        self.deep = _mlp(input_dim, hidden_dims[:-1], hidden_dims[-1]) if hidden_dims else nn.Identity()
        deep_dim = int(hidden_dims[-1]) if hidden_dims else input_dim
        self.output = nn.Linear(input_dim + deep_dim, 1)

    def forward(self, batch: Mapping[str, Tensor]) -> Tensor:
        x0 = self.tokenizer(batch).flatten(start_dim=1)
        crossed = x0
        for layer in self.cross_layers:
            crossed = x0 * layer(crossed) + crossed
        return self.output(torch.cat((crossed, self.deep(x0)), dim=-1)).squeeze(-1)

    @property
    def embedding_parameter_count(self) -> int:
        return self.tokenizer.embedding_parameter_count


class TokenMixing(nn.Module):
    """Parameter-free RankMixer head shuffle, with H equal to token count.

    When D is not divisible by H, heads differ by one channel and the flattened
    result is repartitioned into H equal output tokens. This preserves every
    element and reduces exactly to the paper's shuffle when D is divisible.
    """

    def __init__(self, num_tokens: int, embedding_dim: int) -> None:
        super().__init__()
        if num_tokens < 1 or embedding_dim < 1:
            raise ValueError("num_tokens and embedding_dim must be positive")
        self.num_tokens = num_tokens
        self.embedding_dim = embedding_dim
        positions = torch.arange(num_tokens * embedding_dim).reshape(num_tokens, embedding_dim)
        chunks = torch.tensor_split(positions, num_tokens, dim=1)
        permutation = torch.cat([chunk.reshape(-1) for chunk in chunks])
        self.register_buffer("permutation", permutation, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[1:] != (self.num_tokens, self.embedding_dim):
            raise ValueError("token input must have shape [batch, num_tokens, embedding_dim]")
        return x.flatten(start_dim=1).index_select(1, self.permutation).reshape_as(x)


class Expert(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


class PerTokenFFN(nn.Module):
    """A dedicated dense two-layer FFN for every mixed token."""

    def __init__(self, num_tokens: int, embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.experts = nn.ModuleList(Expert(embedding_dim, hidden_dim) for _ in range(num_tokens))

    def forward(self, x: Tensor) -> Tensor:
        return torch.stack([expert(x[:, t, :]) for t, expert in enumerate(self.experts)], dim=1)


class SparsePerTokenFFN(nn.Module):
    """Separate top-k router and expert pool for each token position.

    Only selected batch rows enter each routed expert in both train and eval.
    The raw softmax gate is scaled by E/k; this keeps task gradients for top-1
    routing and follows the gate-value scaling of TokenMixer-Large.
    """

    def __init__(
        self,
        num_tokens: int,
        embedding_dim: int,
        hidden_dim: int,
        num_experts: int,
        top_k: int,
        shared_expert: bool = False,
    ) -> None:
        super().__init__()
        if num_experts < 2 or not 1 <= top_k <= num_experts:
            raise ValueError("sparse mode requires num_experts >= 2 and 1 <= top_k <= num_experts")
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.top_k = top_k
        self.routers = nn.ModuleList(nn.Linear(embedding_dim, num_experts) for _ in range(num_tokens))
        self.experts = nn.ModuleList(
            nn.ModuleList(Expert(embedding_dim, hidden_dim) for _ in range(num_experts))
            for _ in range(num_tokens)
        )
        self.shared_experts = (
            nn.ModuleList(Expert(embedding_dim, hidden_dim) for _ in range(num_tokens))
            if shared_expert else None
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        outputs = []
        balance_losses = []
        counts = []
        loads = []
        importances = []
        for t, (router, expert_pool) in enumerate(zip(self.routers, self.experts)):
            x_t = x[:, t, :]
            probabilities = F.softmax(router(x_t).float(), dim=-1)
            selected_prob, selected_idx = torch.topk(probabilities, self.top_k, dim=-1)
            selected_weight = selected_prob.to(x.dtype) * (self.num_experts / self.top_k)
            output_t = torch.zeros_like(x_t)
            for expert_id, expert in enumerate(expert_pool):
                matches = torch.nonzero(selected_idx == expert_id, as_tuple=False)
                if matches.numel() == 0:
                    continue
                rows, slots = matches.unbind(dim=1)
                expert_input = x_t.index_select(0, rows)
                contribution = expert(expert_input) * selected_weight[rows, slots].unsqueeze(-1)
                output_t.index_add_(0, rows, contribution)
            if self.shared_experts is not None:
                output_t = output_t + self.shared_experts[t](x_t)
            outputs.append(output_t)

            importance = probabilities.mean(dim=0)
            hard_counts = F.one_hot(selected_idx, self.num_experts).sum(dim=(0, 1))
            hard_load = hard_counts.to(probabilities.dtype) / (x_t.shape[0] * self.top_k)
            balance_losses.append(self.num_experts * torch.sum(importance * hard_load.detach()))
            counts.append(hard_counts.detach())
            loads.append(hard_load.detach())
            importances.append(importance.detach())
        return (
            torch.stack(outputs, dim=1),
            torch.stack(balance_losses).mean(),
            torch.stack(counts),
            torch.stack(loads),
            torch.stack(importances),
        )


class RankMixerBlock(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        embedding_dim: int,
        ffn_hidden_dim: int,
        num_experts: int,
        top_k: int,
        shared_expert: bool,
    ) -> None:
        super().__init__()
        self.mixer = TokenMixing(num_tokens, embedding_dim)
        self.mix_norm = nn.LayerNorm(embedding_dim)
        self.ffn_norm = nn.LayerNorm(embedding_dim)
        self.ffn: nn.Module
        if num_experts == 1:
            self.ffn = PerTokenFFN(num_tokens, embedding_dim, ffn_hidden_dim)
        else:
            self.ffn = SparsePerTokenFFN(
                num_tokens, embedding_dim, ffn_hidden_dim, num_experts, top_k, shared_expert
            )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        mixed = self.mix_norm(self.mixer(x) + x)
        if isinstance(self.ffn, SparsePerTokenFFN):
            transformed, balance, counts, load, importance = self.ffn(mixed)
            stats = {"expert_counts": counts, "expert_load": load, "expert_importance": importance}
        else:
            transformed = self.ffn(mixed)
            balance = x.new_zeros(())
            stats = {}
        return self.ffn_norm(transformed + mixed), balance, stats


class RankMixerRanker(nn.Module):
    """RankMixer CTR ranker; set num_experts > 1 for sparse top-k MoE.

    ``forward_with_aux`` returns unweighted load-balancing loss, so callers can
    set the auxiliary coefficient without coupling it to model construction.
    """

    def __init__(
        self,
        vocab_sizes: Sequence[int],
        num_dense: int,
        embedding_dim: int = 64,
        num_layers: int = 2,
        ffn_hidden_dim: int = 128,
        num_experts: int = 1,
        top_k: int = 1,
        shared_expert: bool = False,
        token_groups: object = None,
    ) -> None:
        super().__init__()
        if num_layers < 1 or ffn_hidden_dim < 1 or num_experts < 1:
            raise ValueError("num_layers, ffn_hidden_dim, and num_experts must be positive")
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        if num_experts == 1 and shared_expert:
            raise ValueError("shared_expert requires num_experts > 1")
        self.tokenizer = FeatureTokenizer(vocab_sizes, num_dense, embedding_dim, token_groups)
        self.num_experts = num_experts
        self.top_k = top_k
        self.blocks = nn.ModuleList(
            RankMixerBlock(
                self.tokenizer.num_tokens,
                embedding_dim,
                ffn_hidden_dim,
                num_experts,
                top_k,
                shared_expert,
            )
            for _ in range(num_layers)
        )
        self.output = nn.Linear(embedding_dim, 1)

    def forward_with_aux(self, batch: Mapping[str, Tensor]) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        x = self.tokenizer(batch)
        losses = []
        counts = []
        load = []
        importance = []
        for block in self.blocks:
            x, balance, stats = block(x)
            losses.append(balance)
            if stats:
                counts.append(stats["expert_counts"])
                load.append(stats["expert_load"])
                importance.append(stats["expert_importance"])
        logits = self.output(x.mean(dim=1)).squeeze(-1)
        aux = torch.stack(losses).mean()
        routing_stats = (
            {
                "expert_counts": torch.stack(counts),
                "expert_load": torch.stack(load),
                "expert_importance": torch.stack(importance),
            }
            if load else {}
        )
        return logits, aux, routing_stats

    def forward(self, batch: Mapping[str, Tensor]) -> Tensor:
        return self.forward_with_aux(batch)[0]

    @property
    def embedding_parameter_count(self) -> int:
        """Number of raw categorical lookup-table parameters."""
        return self.tokenizer.embedding_parameter_count

    def parameter_summary(self) -> dict[str, int]:
        """Report total and per-example active interaction parameters.

        Embedding tables are counted only in ``total_parameters``. Their sparse
        row lookups are not estimated as interaction compute.
        """
        total = sum(p.numel() for p in self.parameters())
        interaction_total = sum(p.numel() for block in self.blocks for p in block.parameters())
        interaction_active = interaction_total
        if self.num_experts > 1:
            for block in self.blocks:
                ffn = block.ffn
                assert isinstance(ffn, SparsePerTokenFFN)
                all_routed = sum(p.numel() for pool in ffn.experts for expert in pool for p in expert.parameters())
                one_expert = sum(p.numel() for p in ffn.experts[0][0].parameters())
                interaction_active -= all_routed - ffn.num_tokens * self.top_k * one_expert
        return {
            "total_parameters": total,
            "interaction_parameters": interaction_total,
            "active_interaction_parameters_per_example": interaction_active,
        }
