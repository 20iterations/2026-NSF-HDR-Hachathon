"""Domain-Adversarial Neural Network (DANN) components.

Reference: Ganin et al., "Domain-Adversarial Training of Neural Networks",
JMLR 2016.

Uses gradient reversal to force the temporal encoder to produce
session-invariant features. A domain classifier tries to predict session ID
from TE hidden states; the gradient reversal layer negates gradients flowing
back to TE, so TE learns to fool the classifier.
"""

import math
import torch
import torch.nn as nn


class GradientReversal(torch.autograd.Function):
    """Gradient Reversal Layer (Ganin et al., JMLR 2016).

    Forward: identity. Backward: negate and scale gradients by alpha.
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class DomainClassifier(nn.Module):
    """Session classifier with gradient reversal for domain-adversarial training.

    A small MLP that predicts session ID from mean-pooled TE hidden states.
    Gradient reversal ensures the encoder learns session-invariant features.
    """

    def __init__(self, hidden_dim: int, num_sessions: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_sessions),
        )

    def forward(self, h: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """Classify session from hidden features with gradient reversal.

        Args:
            h: (B, D) mean-pooled TE hidden states
            alpha: gradient reversal strength (0 at start, ramps to 1)
        Returns:
            logits: (B, num_sessions) session classification logits
        """
        h_rev = GradientReversal.apply(h, alpha)
        return self.classifier(h_rev)


def dann_alpha_schedule(epoch: int, total_epochs: int) -> float:
    """DANN alpha schedule: ramps from 0 to 1 over training.

    Standard schedule from Ganin et al., JMLR 2016.

    Args:
        epoch: current epoch (1-indexed)
        total_epochs: total number of training epochs
    Returns:
        alpha: float in [0, 1]
    """
    p = epoch / total_epochs
    return 2.0 / (1.0 + math.exp(-10 * p)) - 1.0
