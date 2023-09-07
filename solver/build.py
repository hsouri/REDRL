import torch
import pdb


def make_optimizer(cfg, model1, model2=None, model3=None):
    models = [model1]
    if model2 is not None:
        models += [model2]
    if model3 is not None:
        models += [model3]
    params = []
    for model in models:
        for key, value in model.named_parameters():
            if not value.requires_grad:
                continue
            lr = cfg.Base_Learning_Rate
            weight_decay = cfg.Weight_Decay
            if 'bias' in key:
                lr *= cfg.Bias_Learning_Rate_Factor
                weight_decay = cfg.Weight_Decay_Bias
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if cfg.Optimizer == 'SGD':
        optmizer = getattr(torch.optim, cfg.Optimizer)(params, momentum=cfg.Momentum, nesterov=cfg.Nestrov)
    else:
        optmizer = getattr(torch.optim, cfg.Optimizer)(params)
    return optmizer
