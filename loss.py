import torch
import torch.nn as nn

import numpy as np

class SupContrastLoss(nn.Module):
  def __init__(self, temporature):
    super(SupContrastLoss, self).__init__()
    self.temporature = temporature
  def forward(self, features, labels):
    labels = labels.contiguous().view(-1, 1)
    batch_size = features.size(0)
    anchor_features = torch.cat(torch.unbind(features, 1), 0) # [batch_size* view x dim]
    anchor_view = features.size(1)

    global_features = anchor_features
    global_view = anchor_view

    mask = torch.eq(labels, labels.T).float()
    diagonal_mask = torch.eye(batch_size * anchor_view, batch_size * global_view).to('cuda')
    positive_mask = mask.repeat(anchor_view, global_view) - diagonal_mask
    num_positive_row = positive_mask.sum(1, keepdim = True)
    neg_mask = 1 - mask.repeat(anchor_view, global_view)

    logits = torch.matmul(anchor_features, global_features.T) / self.temporature
    max_log, _ = torch.max(logits, 1, keepdim = True)
    logits = logits - max_log.detach()
    exp_log = torch.exp(logits)

    denominator = (exp_log*positive_mask + exp_log*neg_mask).sum(1, keepdim = True)
    loss = (logits - torch.log(denominator)) * positive_mask
    loss = loss.sum(1, keepdim = True)
    loss = - loss / num_positive_row

    loss = loss.view([anchor_view, batch_size]).mean()

    return loss
