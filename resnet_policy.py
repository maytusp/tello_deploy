#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from collections import OrderedDict
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch
from gym import spaces
from torch import nn as nn
from torch.nn import functional as F
from torchvision import transforms as T
from torchvision.transforms import functional as TF

from common.running_mean_and_var import (
    RunningMeanAndVar,
)

class PointNavResNetNet(nn.Module):
    """Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    PRETRAINED_VISUAL_FEATURES_KEY = "visual_features"
    prev_action_embedding: nn.Module

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int,
        num_recurrent_layers: int,
        rnn_type: str,
        backbone,
        resnet_baseplanes,
        normalize_visual_inputs: bool,
        fuse_keys: Optional[List[str]],
        force_blind_policy: bool = False,
        discrete_actions: bool = True,
        norm: str = "batchnorm"
    ):
        print(f"Use norm name {norm}")
        #TODO Make groupnorm and batchnorm be able to select from config not manually select like this
        if norm == "groupnorm":
            import resnet
        elif norm == "batchnorm":
            import resnetbn as resnet
        super().__init__()
        self.prev_action_embedding: nn.Module
        self.discrete_actions = discrete_actions
        self._n_prev_action = 32
        self.num_recurrent_layers = num_recurrent_layers
        if discrete_actions:
            self.prev_action_embedding = nn.Embedding(
                action_space.n + 1, self._n_prev_action
            )
        else:
            num_actions = get_num_actions(action_space)
            self.prev_action_embedding = nn.Linear(
                num_actions, self._n_prev_action
            )
        self._n_prev_action = 32
        rnn_input_size = self._n_prev_action  # test


        self.pointgoal_embedding = nn.Linear(2, 32)
        rnn_input_size += 32


        self._hidden_size = hidden_size

        #TODO Custom obs sapce
        use_obs_space = spaces.Dict(
            {
                k: observation_space.spaces[k]
                for k in fuse_keys
                if len(observation_space.spaces[k].shape) == 3
            }
        )


        self.visual_encoder = ResNetEncoder(
            use_obs_space,
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,
        )


        self.visual_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                np.prod(self.visual_encoder.output_shape), hidden_size
            ),
            nn.ReLU(True),
        )
        # This is the LSTM abstraction from Habitat-lab which is used in DD-PPO training
        # We may not need this abstraction, just use simple LSTM from pytorch
        # self.state_encoder = build_rnn_state_encoder(
        #     (self._hidden_size) + rnn_input_size,
        #     self._hidden_size,
        #     rnn_type=rnn_type,
        #     num_layers=num_recurrent_layers,
        # )

        self.state_encoder = nn.LSTM(
            (self._hidden_size) + rnn_input_size,
            self._hidden_size,
            num_layers=num_recurrent_layers,
        )

        self.train()

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        x = []
        aux_loss_state = {}
        h,c = rnn_hidden_states[0], rnn_hidden_states[1] # Tuple (h,c)
        
        visual_feats = self.visual_encoder(observations)
        visual_feats = self.visual_fc(visual_feats)
        aux_loss_state["perception_embed"] = visual_feats
        x.append(visual_feats)


        goal_observations = observations["point_goal"]
        x.append(self.pointgoal_embedding(goal_observations))
        prev_actions = self.prev_action_embedding(
               prev_actions.float()
            )

        x.append(prev_actions)
        out = torch.cat(x, dim=1)

        out = torch.unsqueeze(out, dim=0)
        out, (h,c) = self.state_encoder(
            out, (h,c) 
        )
        aux_loss_state["rnn_output"] = out

        return out, (h,c), aux_loss_state



class ResNetEncoder(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Dict,
        baseplanes: int = 32,
        ngroups: int = 32,
        spatial_size: int = 128,
        make_backbone=None,
        normalize_visual_inputs: bool = False,
    ):
        super().__init__()

        # Determine which visual observations are present
        self.visual_keys = [
            k
            for k, v in observation_space.spaces.items()
            if len(v.shape) > 1
        ]
        self.key_needs_rescaling = {k: None for k in self.visual_keys}
        for k, v in observation_space.spaces.items():
            if v.dtype == np.uint8:
                self.key_needs_rescaling[k] = 1.0 / v.high.max()

        # Count total # of channels
        self._n_input_channels = sum(
            observation_space.spaces[k].shape[2] for k in self.visual_keys
        )

        if normalize_visual_inputs:
            self.running_mean_and_var: nn.Module = RunningMeanAndVar(
                self._n_input_channels
            )
        else:
            self.running_mean_and_var = nn.Sequential()

        spatial_size_h = (
            observation_space.spaces[self.visual_keys[0]].shape[0] // 2
        )
        spatial_size_w = (
            observation_space.spaces[self.visual_keys[0]].shape[1] // 2
        )
        self.backbone = make_backbone(
            self._n_input_channels, baseplanes, ngroups
        )

        final_spatial_h = int(
            np.ceil(spatial_size_h * self.backbone.final_spatial_compress)
        )
        final_spatial_w = int(
            np.ceil(spatial_size_w * self.backbone.final_spatial_compress)
        )
        after_compression_flat_size = 2048
        num_compression_channels = int(
            round(
                after_compression_flat_size
                / (final_spatial_h * final_spatial_w)
            )
        )
        self.compression = nn.Sequential(
            nn.Conv2d(
                self.backbone.final_channels,
                num_compression_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(1, num_compression_channels), # This is IN because num_groups=1
            nn.ReLU(True),
        )

        self.output_shape = (
            num_compression_channels,
            final_spatial_h,
            final_spatial_w,
        )


    def layer_init(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:  # type: ignore
        cnn_input = []
        for k in self.visual_keys:
            obs_k = observations[k]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            obs_k = obs_k.permute(0, 3, 1, 2)
            if self.key_needs_rescaling[k] is not None:
                obs_k = (
                    obs_k.float() * self.key_needs_rescaling[k]
                )  # normalize
            cnn_input.append(obs_k)

        x = torch.cat(cnn_input, dim=1)
        # print("before avg pool", x.shape)
        x = F.avg_pool2d(x, 2)
        # print("after avg pool", x.shape)
        x = self.running_mean_and_var(x)
        x = self.backbone(x)
        # print("after backbone", x.shape)
        x = self.compression(x)
        # print("after compression", x.shape)

        return x

def get_num_actions(action_space) -> int:
    num_actions = 0
    for v in iterate_action_space_recursively(action_space):
        if isinstance(v, spaces.Box):
            assert (
                len(v.shape) == 1
            ), f"shape was {v.shape} but was expecting a 1D action"
            num_actions += v.shape[0]
        elif isinstance(v, EmptySpace):
            num_actions += 1
        elif isinstance(v, spaces.Discrete):
            num_actions += v.n
        else:
            raise NotImplementedError(
                f"Trying to count the number of actions with an unknown action space {v}"
            )

    return num_actions

def iterate_action_space_recursively(action_space):
    if isinstance(action_space, spaces.Dict):
        for v in action_space.values():
            yield from iterate_action_space_recursively(v)
    else:
        yield action_space
