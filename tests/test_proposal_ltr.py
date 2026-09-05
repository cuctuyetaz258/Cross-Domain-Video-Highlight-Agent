import torch

from highlight_agent.models.actionformer import TemporalProposal
from highlight_agent.models.proposal_ltr import (
    ContextAwareProposalLTRScorer,
    ProposalLTRConfig,
    ProposalLTRScorer,
    pairwise_proposal_loss,
)


def test_proposal_ltr_pools_context_and_scores() -> None:
    model = ProposalLTRScorer(8, hidden_dim=16, dropout=0.0)
    features = torch.rand(1, 8, 100)
    proposals = [[TemporalProposal(10, 40, 0.8, 0, 20)]]

    scores, provenance = model(features, proposals, stride_seconds=0.5)

    assert scores.shape == (1,)
    assert provenance == [(0, 0)]
    assert torch.isfinite(scores).all()


def test_pairwise_proposal_loss_uses_only_within_video_pairs() -> None:
    scores = torch.tensor([0.0, 1.0, 5.0, -5.0], requires_grad=True)
    utilities = torch.tensor([1.0, 0.0, 0.0, 1.0])
    videos = torch.tensor([0, 0, 1, 1])

    loss = pairwise_proposal_loss(
        scores,
        utilities,
        videos,
        margin=1.0,
        utility_delta=0.5,
    )

    assert loss.item() == 6.5
    loss.backward()
    assert scores.grad is not None


def _imsab_scorer() -> ContextAwareProposalLTRScorer:
    return ContextAwareProposalLTRScorer(
        8,
        config=ProposalLTRConfig(
            d_model=16,
            num_imsab_blocks=2,
            num_inducing_points=4,
            num_heads=2,
            ffn_dim=32,
            dropout=0.0,
        ),
    )


def test_imsab_scorer_is_permutation_equivariant() -> None:
    torch.manual_seed(7)
    model = _imsab_scorer().eval()
    features = torch.rand(1, 8, 200)
    proposals = [
        TemporalProposal(5, 35, 0.7, 0, 10),
        TemporalProposal(40, 80, 0.9, 1, 20),
        TemporalProposal(90, 150, 0.4, 0, 30),
    ]

    original, _ = model(features, [proposals], stride_seconds=1.0)
    permutation = [2, 0, 1]
    shuffled, _ = model(
        features,
        [[proposals[index] for index in permutation]],
        stride_seconds=1.0,
    )

    assert torch.allclose(shuffled, original[permutation], atol=1e-5)


def test_imsab_padding_does_not_change_valid_scores() -> None:
    torch.manual_seed(11)
    model = _imsab_scorer().eval()
    features = torch.rand(2, 8, 200)
    first = [TemporalProposal(10, 40, 0.8, 0, 20)]
    second = [
        TemporalProposal(0, 30, 0.6, 0, 10),
        TemporalProposal(40, 80, 0.7, 0, 30),
        TemporalProposal(100, 160, 0.9, 1, 50),
    ]

    alone, _ = model(features[:1], [first], stride_seconds=1.0)
    batched, provenance = model(features, [first, second], stride_seconds=1.0)

    assert provenance[0] == (0, 0)
    assert torch.allclose(alone, batched[:1], atol=1e-5)


def test_imsab_inducing_points_receive_gradients() -> None:
    model = _imsab_scorer()
    features = torch.rand(1, 8, 120)
    proposals = [[
        TemporalProposal(0, 30, 0.8, 0, 10),
        TemporalProposal(40, 90, 0.5, 1, 20),
    ]]

    scores, _ = model(features, proposals, stride_seconds=1.0)
    scores.sum().backward()

    assert model.blocks[0].inducing_points.grad is not None
    assert torch.isfinite(model.blocks[0].inducing_points.grad).all()
