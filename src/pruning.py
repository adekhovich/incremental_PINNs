import torch
import torch
import torch.nn as nn
from torch.nn import functional as F


import numpy as np



def mlp_total_params(model):
    total_number = 0
    for param_name in list(model.state_dict()):
        param = model.state_dict()[param_name]
        total_number += torch.numel(param[param != 0])

    return total_number


def mlp_total_params_mask(model, task_id):
    total_number_fc = torch.tensor(0, dtype=torch.int32)
    for mask in model.tasks_masks[task_id]:
        total_number_fc += mask.sum().int()


    return total_number_fc.item()


def mlp_fc_pruning(net, alpha, x_batch, task_id, device, start_prune=0):
    layers = list(net.state_dict())

    num_samples = x_batch.size()[0]
   
    for l in range(0, len(layers)//2):

        fc_weight = net.state_dict()[layers[2*l]]*net.tasks_masks[task_id][2*l].to(device)
        fc_bias = net.state_dict()[layers[2*l+1]]*net.tasks_masks[task_id][2*l+1].to(device)

        curr_layer = F.linear(x_batch, weight=fc_weight, bias=fc_bias)

        if 2*l+1 < len(list(net.named_children())):
            activation = lambda x: net.activation
        else:
            activation = lambda x: x

        for i in range(curr_layer.size()[1]):
            avg_neuron_val = torch.mean(activation(curr_layer), axis=0)[i]

            if (avg_neuron_val == 0):
                net.tasks_masks[task_id][2*l][i] = 0
                net.tasks_masks[task_id][2*l+1][i] = 0
            else:
                flow = torch.cat((x_batch*fc_weight[i], torch.reshape(fc_bias[i].repeat(num_samples), (-1, 1))), dim=1).abs()

                importances = torch.mean(torch.abs(flow), dim=0)

                sum_importance = torch.sum(importances)
                sorted_importances, sorted_indices = torch.sort(importances, descending=True)

                cumsum_importances = torch.cumsum(importances[sorted_indices], dim=0)
                pivot = torch.sum(cumsum_importances < alpha*sum_importance)

                if pivot < importances.size(0) - 1:
                    pivot += 1
                else:
                    pivot = importances.size(0) - 1

                thresh = importances[sorted_indices][pivot]

                net.tasks_masks[task_id][2*l][i][importances[:-1] <= thresh] = 0

                if importances[-1] <= thresh:
                    net.tasks_masks[task_id][2*l+1][i] = 0

    
        if 2*l+1 < len(layers):
            x_batch = activation(curr_layer)


    return net


def mlp_backward_pruning(net, task_id):
    h = len(list(net.state_dict()))//2-1
    while h > 0:
        pruned_neurons = torch.nonzero( net.tasks_masks[task_id][2*h].sum(dim=0) == 0).reshape(1, -1).squeeze(0)
        net.tasks_masks[task_id][2*(h-1)][pruned_neurons] = 0
        net.tasks_masks[task_id][2*(h-1)+1][pruned_neurons] = 0

        h -= 1

    return net


def mlp_pruning(net, alpha_fc, x_batch, task_id, device, start_fc_prune=0):
    
    net = mlp_fc_pruning(net, alpha_fc, x_batch, task_id, device, start_fc_prune)

    net = mlp_backward_pruning(net, task_id)

    return net
