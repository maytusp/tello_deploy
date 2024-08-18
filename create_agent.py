
from resnet_policy import *
from constants import *
import dua

import gym
from gym import spaces
import numpy as np
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class NetPolicy(nn.Module):
    def __init__(self, 
            pretrained_enc_path=None, 
            use_dua=False,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.net = PointNavResNetNet(observation_space=observation_space,
                action_space=action_space,
                hidden_size=512,
                num_recurrent_layers=2,
                rnn_type="lstm",
                backbone="se_resneXt50",
                fuse_keys=["rgb_image"],
                resnet_baseplanes = 32,
                normalize_visual_inputs = True,
                force_blind_policy = False,
                discrete_actions = False,
                norm = "batchnorm").to(device)

        self.action_dims = action_space.shape[0]
        self.action_head = nn.Linear(self.net._hidden_size, self.action_dims).to(device)

        ### Only for deploying on real robots or single simulation
        self.h_test = torch.randn(self.net.num_recurrent_layers, 1, self.net._hidden_size).to(device)
        self.c_test = torch.randn(self.net.num_recurrent_layers, 1, self.net._hidden_size).to(device)
        self.prev_actions_test = torch.zeros(1, self.action_dims).to(device)

        ### DUA Test-time adaptation (only works for BatchNorm-based Visual Encoder (CNN)###
        self.use_dua = use_dua
        if self.use_dua:
            print("Use DUA adaptation")
            self.mom_pre= 0.1
            self.decay_factor = 0.94
            self.min_mom = 0.005
        ###

        if pretrained_enc_path is not None:
            prefix = "net.visual_encoder."
            pretrained_state_dict = torch.load(pretrained_enc_path)
            self.net.visual_encoder.load_state_dict(
                {
                    k[len(prefix) :]: v
                    for k, v in pretrained_state_dict.items()
                    if k.startswith(prefix)
                }
            )

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
    ):
        features, rnn_hidden_states, _  = self.net(observations, rnn_hidden_states, prev_actions)
        action = torch.squeeze(self.action_head(features))
        return action, rnn_hidden_states

    # For deploying on real robot or single simulation
    def get_action(
        self,
        observations
    ):
        with torch.no_grad():
            # If use DUA
            if self.use_dua:
                self.net, self.mom_pre, self.decay_factor, self.min_mom = dua._adapt(self.net, 
                                                                                    self.mom_pre, 
                                                                                    self.decay_factor, 
                                                                                    self.min_mom)
                # Update Running statistics
                _, _ , _  = self.net(observations, (self.h_test, self.c_test), self.prev_actions_test)
                self.net.eval() # BatchNorm.eval() to use running statistics in normalization

            # Main inference
            features, (h,c) , _  = self.net(observations, (self.h_test, self.c_test), self.prev_actions_test)
            action = self.action_head(features)
            self.h_test, self.c_test = h, c
            self.prev_actions_test = torch.squeeze(action.detach(), dim=1)

        return torch.squeeze(action) # (action_dims,)
        


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # prev_actions = torch.tensor(np.random.randn(batch_size, 4), dtype=torch.float32)
    # masks = torch.tensor([[True], [True], [True], [True], [False]])
    # rnn_hidden_states = torch.tensor(np.random.randn(batch_size,4,512), dtype=torch.float32)
    agent = NetPolicy(use_dua=True,pretrained_enc_path="checkpoints/ckpt_20000.pth")
    agent.to(device)
    for i in range(10):
        observations = dict()
        observations["rgb_image"] = torch.tensor(np.random.randn(1,256,256,3), dtype=torch.float32).to(device)
        observations["point_goal"] = torch.tensor(np.random.randn(1,2), dtype=torch.float32).to(device)
        action = agent.get_action(observations)
        # print(i, action)