import torch

def _adapt(agent,
            mom_pre,
            decay_factor,
            min_mom):
    encoder = agent.visual_encoder
    mom_new = (mom_pre * decay_factor)
    min_momentum_constant = min_mom
    for m in encoder.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.train()
            m.momentum = mom_new + min_momentum_constant
    mom_pre = mom_new
    return agent, mom_pre, decay_factor, min_mom