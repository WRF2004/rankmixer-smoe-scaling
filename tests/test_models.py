import pytest
import torch

from rmscaling.models import (
    DCNv2Ranker,
    FeatureTokenizer,
    MLPRanker,
    RankMixerRanker,
    SparsePerTokenFFN,
    TokenMixing,
)


def batch(batch_size=5):
    return {
        "categorical": torch.tensor([[i % 7, (i + 1) % 5, i % 3] for i in range(batch_size)]),
        "numerical": torch.linspace(-1, 1, batch_size).unsqueeze(1),
        "label": torch.tensor([i % 2 for i in range(batch_size)], dtype=torch.float32),
    }


def test_token_mixing_matches_paper_shuffle_when_divisible():
    x = torch.arange(2 * 4 * 8, dtype=torch.float32).reshape(2, 4, 8)
    expected = x.reshape(2, 4, 4, 2).transpose(1, 2).reshape(2, 4, 8)
    torch.testing.assert_close(TokenMixing(4, 8)(x), expected)


def test_token_mixing_preserves_all_channels_when_not_divisible():
    x = torch.arange(3 * 5, dtype=torch.float32).reshape(1, 3, 5)
    output = TokenMixing(3, 5)(x)
    assert output.shape == x.shape
    torch.testing.assert_close(output.flatten().sort().values, x.flatten().sort().values)


def test_tokenizer_semantic_groups_partition_fields():
    tok = FeatureTokenizer((7, 5, 3), 1, 8, token_groups={"user": (0, 1), "item": (2,)})
    assert tok(batch()).shape == (5, 3, 8)
    with pytest.raises(ValueError, match="partition"):
        FeatureTokenizer((7, 5, 3), 1, 8, token_groups=((0, 1), (1, 2)))


@pytest.mark.parametrize("ranker", [MLPRanker, DCNv2Ranker])
def test_baselines_produce_logits_and_gradients(ranker):
    torch.manual_seed(11)
    model = ranker((7, 5, 3), 1, embedding_dim=8, hidden_dims=(16, 8))
    logits = model(batch())
    assert logits.shape == (5,)
    assert torch.isfinite(logits).all()
    logits.sum().backward()
    assert any(p.grad is not None for p in model.parameters())
    assert model.embedding_parameter_count == (7 + 5 + 3) * 8


def test_sparse_dispatch_only_calls_selected_experts_and_shared_receives_all_rows():
    torch.manual_seed(12)
    layer = SparsePerTokenFFN(2, 4, 6, num_experts=3, top_k=1, shared_expert=True)
    with torch.no_grad():
        for router in layer.routers:
            router.weight.zero_()
            router.bias.copy_(torch.tensor([0.0, 0.0, -10.0]))
            router.weight[0, 0] = 1.0
            router.weight[1, 0] = -1.0
    x = torch.tensor(
        [[[2.0, 0, 0, 0], [2.0, 0, 0, 0]],
         [[-2.0, 0, 0, 0], [-2.0, 0, 0, 0]],
         [[3.0, 0, 0, 0], [3.0, 0, 0, 0]],
         [[-3.0, 0, 0, 0], [-3.0, 0, 0, 0]]],
        requires_grad=True,
    )
    called = {}
    hooks = []
    for t, pool in enumerate(layer.experts):
        for e, expert in enumerate(pool):
            hooks.append(expert.register_forward_hook(
                lambda module, inputs, output, t=t, e=e: called.__setitem__((t, e), inputs[0].shape[0])
            ))
    for t, expert in enumerate(layer.shared_experts):
        hooks.append(expert.register_forward_hook(
            lambda module, inputs, output, t=t: called.__setitem__((t, "shared"), inputs[0].shape[0])
        ))
    output, balance, counts, load, importance = layer(x)
    for hook in hooks:
        hook.remove()
    assert output.shape == x.shape
    assert counts.tolist() == [[2, 2, 0], [2, 2, 0]]
    assert called == {
        (0, 0): 2, (0, 1): 2, (0, "shared"): 4,
        (1, 0): 2, (1, 1): 2, (1, "shared"): 4,
    }
    torch.testing.assert_close(load.sum(-1), torch.ones(2))
    torch.testing.assert_close(importance.sum(-1), torch.ones(2))
    (output.sum() + 0.01 * balance).backward()
    assert x.grad is not None
    assert all(router.weight.grad is not None for router in layer.routers)
    assert all(pool[2].network[0].weight.grad is None for pool in layer.experts)


@pytest.mark.parametrize("top_k", [1, 2])
def test_rankmixer_sparse_aux_and_parameter_accounting(top_k):
    torch.manual_seed(13)
    model = RankMixerRanker(
        (7, 5, 3), 1, embedding_dim=8, num_layers=2,
        ffn_hidden_dim=16, num_experts=4, top_k=top_k,
    )
    logits, aux, stats = model.forward_with_aux(batch())
    assert logits.shape == (5,)
    assert aux.ndim == 0 and torch.isfinite(aux) and aux > 0
    assert stats["expert_counts"].shape == (2, 4, 4)
    assert torch.all(stats["expert_counts"].sum(-1) == 5 * top_k)
    assert stats["expert_load"].shape == (2, 4, 4)
    assert stats["expert_importance"].shape == (2, 4, 4)
    summary = model.parameter_summary()
    assert summary["active_interaction_parameters_per_example"] < summary["interaction_parameters"]
    assert summary["total_parameters"] >= summary["interaction_parameters"]
    assert model.embedding_parameter_count == (7 + 5 + 3) * 8
    (logits.sum() + 0.01 * aux).backward()
    assert any(p.grad is not None for p in model.blocks[0].ffn.routers[0].parameters())


def test_dense_rankmixer_has_zero_aux_and_all_interaction_parameters_active():
    model = RankMixerRanker((7, 5, 3), 1, embedding_dim=8, num_layers=2, ffn_hidden_dim=16)
    logits, aux, stats = model.forward_with_aux(batch())
    assert logits.shape == (5,)
    assert aux.item() == 0.0
    assert stats == {}
    summary = model.parameter_summary()
    assert summary["interaction_parameters"] == summary["active_interaction_parameters_per_example"]
    assert torch.equal(logits, model(batch()))


def test_rankmixer_rejects_invalid_routing():
    with pytest.raises(ValueError, match="top_k"):
        RankMixerRanker((3,), 0, num_experts=2, top_k=3)
