# coding: utf-8
"""
ReLUFormer attention normalization.

Implements the variance-corrected ReLU attention proposed in
"A Study on ReLU and Softmax in Transformer" (Shen et al., 2023),
https://arxiv.org/abs/2302.06461 (Section 4, Eq. 5-7).

Plain Softmax attention is replaced with:

    s_ij = ReLU(q_i^T k_j) / (gamma * sqrt(n_i / 2))                (Eq. 6)

where:
    - n_i is the number of *unmasked* key/value slots visible to
      query position i. This is dynamic: for encoder self-attention
      and encoder-decoder (cross) attention it is simply the (padded)
      source length, but for causal decoder self-attention every
      token i can only see the first i+1 positions, so n_i grows with
      i (this is exactly the "assign different lengths to each token"
      solution described in Section 4.2 of the paper for adapting
      ReLUFormer to the decoder).
    - gamma is a learnable, per-head scale (the paper's "variance
      reduction factor" gamma * sqrt(n/2), Eq. 6). We keep gamma in
      log-space (`log_gamma`) and exponentiate it so it always stays
      positive, and initialize it to 1 (i.e. log_gamma = 0), matching
      "no extra scaling beyond the theoretical sqrt(n/2) term".

Naive ReLU attention collapses onto very few keys (Table 3 in the
paper: ReLU entropy H(s) = 3.45 vs Softmax 1.40, with 94% of the
weights exactly zero), so we additionally compute the regularization
loss from Eq. 7:

    L_reg = | log( sum_j s_ij ) | + max( H(s_i) - C_i , 0 )

- the normalization term pulls the (un-normalized) row sum of weights
  towards 1, which is what pushes ReLUFormer to actually spread
  weight across more than a handful of slots;
- the entropy-margin term keeps the entropy of the row-normalized
  weights from exceeding C_i = entropy_margin_coeff * log(n_i)
  (entropy_margin_coeff = 0.7 in the paper's Appendix A), preventing
  the opposite failure mode of an overly flat/uninformative
  distribution.

This loss is only meaningful during training, so it is only computed
when `self.training` is True; at eval/inference time
`self.last_reg_loss` is set to `None`. Callers (see
`MultiHeadedAttention` in `transformer_layers.py`) are expected to
read `self.last_reg_loss` after `forward()` and, if training,
accumulate it (scaled by a small weight) into the total loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ReLUFormerAttention(nn.Module):
    """
    Variance-corrected ReLU attention normalization ("ReLUFormer").
    Drop-in replacement for `nn.Softmax(dim=-1)` applied to raw
    attention scores (Q K^T / sqrt(d)) inside multi-head attention.
    """

    def __init__(
        self,
        num_heads: int,
        entropy_margin_coeff: float = 0.7,
        eps: float = 1e-8,
    ):
        """
        :param num_heads: number of attention heads; one gamma
            (variance reduction factor) is learned per head
        :param entropy_margin_coeff: the "c" constant in the entropy
            upper bound C = c * log(n) from Appendix A of the paper
            (0.7 there)
        :param eps: small constant used for numerical stability in
            logs/divisions
        """
        super().__init__()
        self.log_gamma = nn.Parameter(torch.zeros(num_heads))
        self.entropy_margin_coeff = entropy_margin_coeff
        self.eps = eps
        self.last_reg_loss = None

    @staticmethod
    def _valid_counts(
        mask: Tensor, batch_size: int, q_len: int, k_len: int, device, dtype
    ) -> Tensor:
        """
        Infer the number of valid (unmasked) key slots visible to
        each query position from the boolean mask.

        :param mask: mask tensor *before* the head dimension is added.
            Either shape (batch, 1, k_len) - the same key mask is
            used for every query position (encoder self-attention,
            encoder-decoder attention), or shape (batch, q_len, k_len)
            - a distinct mask per query position (causal decoder
            self-attention, where the mask already encodes both
            padding and the subsequent-position mask, see
            `TransformerDecoder.forward`).
        :return: tensor of shape (batch, q_len), the number of valid
            key positions for every query position
        """
        if mask is None:
            return torch.full((batch_size, q_len), k_len, device=device, dtype=dtype)
        counts = mask.sum(dim=-1).to(dtype)  # (batch, 1) or (batch, q_len)
        if counts.size(1) == 1 and q_len > 1:
            counts = counts.expand(-1, q_len)
        return counts

    # pylint: disable=arguments-differ
    def forward(self, scores: Tensor, mask: Tensor = None) -> Tensor:
        """
        :param scores: raw attention scores (already scaled by
            1/sqrt(head_size)), shape (batch, num_heads, q_len, k_len)
        :param mask: boolean mask, True/1 at valid positions, shape
            (batch, 1, k_len) or (batch, q_len, k_len) - same masking
            convention used for Softmax attention elsewhere in this
            codebase
        :return: attention weights, shape (batch, num_heads, q_len, k_len)
        """
        batch_size, num_heads, q_len, k_len = scores.shape

        relu_scores = F.relu(scores)
        if mask is not None:
            relu_scores = relu_scores.masked_fill(~mask.unsqueeze(1), 0.0)

        n = self._valid_counts(mask, batch_size, q_len, k_len, scores.device, scores.dtype)
        n = n.clamp(min=1.0)  # avoid division by zero for fully-padded rows

        gamma = torch.exp(self.log_gamma).view(1, num_heads, 1, 1)
        # n has shape (mask_batch, q_len), where mask_batch is either 1
        # (e.g. the dummy causal-only mask used during greedy/beam-search
        # decoding, see search.py) or batch_size. Use unsqueeze + implicit
        # broadcasting here instead of a hard `.view(batch_size, ...)`,
        # since the latter requires exactly batch_size * q_len elements
        # and crashes whenever mask_batch == 1 != batch_size.
        denom = gamma * torch.sqrt(n / 2.0).unsqueeze(1).unsqueeze(-1)
        weights = relu_scores / denom

        if self.training:
            sum_w = weights.sum(dim=-1)  # (batch, heads, q_len)
            norm_reg = torch.log(sum_w + self.eps).abs().mean()

            probs = weights / (sum_w.unsqueeze(-1) + self.eps)
            entropy = -(probs * torch.log(probs + self.eps)).sum(dim=-1)  # (batch, heads, q_len)

            c_bound = self.entropy_margin_coeff * torch.log(n.clamp(min=2.0))
            c_bound = c_bound.unsqueeze(1)  # (batch, 1, q_len), broadcasts over heads

            entropy_reg = F.relu(entropy - c_bound).mean()
            self.last_reg_loss = norm_reg + entropy_reg
        else:
            self.last_reg_loss = None

        return weights

    def __repr__(self):
        return "ReLUFormerAttention(num_heads=%d)" % self.log_gamma.numel()