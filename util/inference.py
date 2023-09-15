"""Run a population through a pre-trained model from disk"""

from util.tools import normalize, center_markers
from util.feature_selection import feature_reduce
import torch
import numpy as np
from copy import deepcopy


def forward(train_x, train_y, eval_x, model, continuous=True):
    train_x = deepcopy(train_x)
    train_y = deepcopy(train_y)
    eval_x = deepcopy(eval_x)

    # Normalize labels to zero mean and unit variance
    train_y = normalize(train_y)

    # Normalize markers to [0.5, 0.5]
    train_x = center_markers(train_x)
    eval_x = center_markers(eval_x)

    # Grab information from model
    device = next(model.parameters()).device
    bucket_means = model.bucket_means
    feature_length = model.feature_length
    feature_selection = model.feature_selection

    eval_start = train_x.shape[0]

    # Perform feature selection
    train_x, eval_x = feature_reduce(train_x, train_y, feature_length, method=feature_selection, eval_x=eval_x)

    train_x = torch.as_tensor(train_x, device=device, dtype=torch.float32)
    train_y = torch.as_tensor(train_y, device=device, dtype=torch.float32)
    eval_x = torch.as_tensor(eval_x, device=device, dtype=torch.float32)

    x = torch.cat((train_x, eval_x), 0)
    x = torch.unsqueeze(x, 0)
    x = x.transpose(1, 0)

    y = torch.cat((train_y, torch.zeros(eval_x.shape[0]).to(device)), 0)
    y = torch.unsqueeze(y, 0)
    y = y.transpose(1, 0)

    out = model(x, y, eval_start)

    if continuous:
        if hasattr(model, 'loss_object'):
            pred = np.squeeze(model.loss_object.mean(out[eval_start:]).detach().cpu().numpy(), -1)
        else:
            sm = torch.nn.functional.softmax(out[eval_start:], dim=2)
            pred = np.sum(np.squeeze(sm.detach().cpu().numpy(), 1) * bucket_means, axis=1)
    else:
        pred = bucket_means[np.argmax(out[eval_start:].detach().cpu().numpy(), axis=2).flatten()]

    return pred
