
# -----------------------------------------------------------------------------
# Original file created on Fri Jun 7 2020
# Original author: Satish
#
# Modifications and new content copyright (c) Yuan Feng, 2025
#
# License: MIT License
# -----------------------------------------------------------------------------

import numpy as np
import os
import sys
import argparse
import random
import torch
import torch.nn as nn
from torch.autograd import Variable, gradcheck


def pearson_correlation(x, y, eps=1e-8):
    mean_x = torch.mean(x, dim=-1, keepdim=True)
    mean_y = torch.mean(y, dim=-1, keepdim=True)

    xm = x - mean_x
    ym = y - mean_y

    # Dot product
    r_num = torch.sum(torch.mul(xm, ym), dim=-1, keepdim=True)
    r_den = (
        torch.norm(xm, 2, dim=-1, keepdim=True)
        * torch.norm(ym, 2, dim=-1, keepdim=True)
    )

    # Avoid division by zero
    r_den = torch.where(r_den == 0, torch.ones_like(r_den), r_den)

    r_val = r_num / r_den
    return r_val


def get_pearson_correlation_vector(x_in, chunk_size=2):
    """
    Calculate pairwise Pearson correlations between channels.

    The computation is split into chunks to reduce temporary memory usage.
    The output order is preserved to match the original implementation.

    Args:
        x_in: Tensor with shape [batch, channels, time_steps].
        chunk_size: Number of channel rows processed in each chunk.

    Returns:
        Tensor with shape [batch, channels * channels].
    """
    if x_in.ndim != 3:
        raise ValueError(
            "x_in must have shape [batch, channels, time_steps]"
        )

    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")

    batch_size, channel_count, time_steps = x_in.shape
    correlation_chunks = []

    for start in range(0, channel_count, chunk_size):
        end = min(start + chunk_size, channel_count)
        current_chunk_size = end - start

        # Select a chunk of the first channel dimension.
        x1 = (
            x_in[:, start:end, :]
            .unsqueeze(2)
            .expand(
                batch_size,
                current_chunk_size,
                channel_count,
                time_steps,
            )
            .reshape(batch_size, -1, time_steps)
        )

        # Compare each selected channel against every channel.
        x2 = (
            x_in.unsqueeze(1)
            .expand(
                batch_size,
                current_chunk_size,
                channel_count,
                time_steps,
            )
            .reshape(batch_size, -1, time_steps)
        )

        chunk_output = pearson_correlation(x1, x2)
        correlation_chunks.append(chunk_output.squeeze(-1))

    return torch.cat(correlation_chunks, dim=1)


def top_k_channel_pairs_choose(
    c_vector,
    ch_num,
    keep_rate=0.5,
    keep_self_cycles=True,
    device="cpu",
):
    assert 0 < keep_rate <= 1.0

    c_mat = c_vector.reshape(c_vector.shape[0], ch_num, ch_num)

    if not keep_self_cycles:
        c_mat = torch.diagonal_scatter(
            c_mat,
            -2 * torch.ones(c_vector.shape[0], ch_num).to(device=device),
            dim1=-2,
            dim2=-1,
        )

    sort_mask = make_tri_mask_2d(
        dim=ch_num,
        diagonal=0,
        type="l",
        device=device,
    ).unsqueeze(0)

    tril_filled_c_mat = c_mat.masked_fill(sort_mask, float(-2))

    num_channel_keep = int(keep_rate * ch_num * (ch_num - 1) / 2)

    tril_filled_flatten = (
        tril_filled_c_mat.flatten(start_dim=-2, end_dim=-1).clone()
    )

    channel_sorted_value, channel_index = torch.sort(
        tril_filled_flatten,
        dim=-1,
        descending=True,
    )

    channel_inf_index = channel_index[:, num_channel_keep:]

    tril_filled_flatten = tril_filled_flatten.scatter(
        -1,
        channel_inf_index,
        float(-2),
    )

    tril_filled_mat = tril_filled_flatten.reshape(-1, ch_num, ch_num)
    tril_filled_mat = repair_tril(tril_filled_mat)

    if keep_self_cycles:
        tril_filled_mat = torch.diagonal_scatter(
            tril_filled_mat,
            torch.ones(c_vector.shape[0], ch_num).to(device=device),
            dim1=-2,
            dim2=-1,
        )
        units_kept = 2 * num_channel_keep + ch_num
    else:
        units_kept = 2 * num_channel_keep

    index_vector = torch.nonzero(tril_filled_mat >= -1)
    index_vector = index_vector.reshape(-1, units_kept, 3)

    return index_vector


def make_tri_mask_2d(dim, diagonal, type="l", device="cpu"):
    if type == "l":
        mask = torch.tril(
            torch.ones((dim, dim), dtype=torch.bool),
            diagonal=diagonal,
        ).to(device)
    elif type == "u":
        mask = torch.triu(
            torch.ones((dim, dim), dtype=torch.bool),
            diagonal=diagonal,
        ).to(device)
    else:
        raise ValueError("type must be 'l' or 'u'")

    return mask


def repair_tril(c_mat):
    result_c_mat = c_mat.clone()
    num_channel = result_c_mat.shape[1]

    for i in range(num_channel - 1):
        for j in range(i + 1, num_channel):
            result_c_mat[:, j, i] = c_mat[:, i, j]

    return result_c_mat
