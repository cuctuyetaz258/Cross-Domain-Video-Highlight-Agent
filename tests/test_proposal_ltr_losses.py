import torch

from highlight_agent.models.proposal_ltr_losses import ranknet_proposal_loss


def test_ranknet_prefers_correct_ordering() -> None:
    utilities = torch.tensor([1.0, 0.5, 0.0])
    videos = torch.zeros(3, dtype=torch.long)

    correct = ranknet_proposal_loss(
        torch.tensor([2.0, 1.0, 0.0]), utilities, videos, weighting="utility"
    )
    reversed_loss = ranknet_proposal_loss(
        torch.tensor([0.0, 1.0, 2.0]), utilities, videos, weighting="utility"
    )

    assert correct < reversed_loss


def test_ranknet_never_builds_cross_video_pairs() -> None:
    scores = torch.tensor([0.0, 10.0], requires_grad=True)
    utilities = torch.tensor([1.0, 0.0])
    videos = torch.tensor([0, 1])

    loss = ranknet_proposal_loss(scores, utilities, videos)

    assert loss.item() == 0.0
    loss.backward()
    assert torch.equal(scores.grad, torch.zeros_like(scores))


def test_delta_ndcg_weighting_is_finite_for_zero_utility() -> None:
    scores = torch.tensor([0.2, 0.1], requires_grad=True)
    utilities = torch.zeros(2)
    videos = torch.zeros(2, dtype=torch.long)

    loss = ranknet_proposal_loss(
        scores,
        utilities,
        videos,
        utility_delta=0.0,
        weighting="delta_ndcg",
    )

    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(scores.grad).all()
