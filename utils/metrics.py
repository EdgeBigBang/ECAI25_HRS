import numpy as np
import torch


def MAE(pred, true):
    return np.mean(np.abs(pred - true))

def MSE(pred, true):
    return np.mean((pred - true) ** 2)

def scheduling_aware_loss(pred, true):
    diff = pred - true
    loss = torch.zeros_like(diff)

    mask_diff_neg = diff < 0
    loss[mask_diff_neg] = torch.abs(diff[mask_diff_neg]) * 0.0065 + 4

    mask_diff_pos = diff >= 0
    loss[mask_diff_pos] = diff[mask_diff_pos] * 0.0035

    return torch.mean(loss)

def average_profit_loss(pred, true):
    diff = pred - true
    loss = np.zeros_like(diff)

    mask_diff_neg = diff < 0
    loss[mask_diff_neg] = np.abs(diff[mask_diff_neg]) * 0.0065 + 4

    mask_diff_pos = diff > 0
    loss[mask_diff_pos] = diff[mask_diff_pos] * 0.0035

    return np.mean(loss)

def metric(pred, true):
    apl = average_profit_loss(pred, true)
    return apl
