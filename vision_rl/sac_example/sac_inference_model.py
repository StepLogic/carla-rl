#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

def get_activation_fn(name=None):
    if name in ["linear", None]:
        return None
    if name == "relu":
        return nn.ReLU
    elif name == "tanh":
        return nn.Tanh
    raise ValueError("Unknown activation ({})!".format(name))

class SlimConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, padding,
                 initializer="default", activation_fn="default", bias_init=0):
        super(SlimConv2d, self).__init__()
        layers = []

        if padding:
            layers.append(nn.ZeroPad2d(padding))

        conv = nn.Conv2d(in_channels, out_channels, kernel, stride)
        if initializer:
            if initializer == "default":
                initializer = nn.init.xavier_uniform_
            initializer(conv.weight)
        nn.init.constant_(conv.bias, bias_init)
        layers.append(conv)

        if isinstance(activation_fn, str):
            activation_fn = get_activation_fn(activation_fn)
        if activation_fn is not None:
            layers.append(activation_fn())

        self._model = nn.Sequential(*layers)

    def forward(self, x):
        return self._model(x)

class SlimFC(nn.Module):
    def __init__(self, in_size, out_size, initializer=None, activation_fn=None, use_bias=True, bias_init=0.0):
        super(SlimFC, self).__init__()
        layers = []

        linear = nn.Linear(in_size, out_size, bias=use_bias)
        if initializer:
            initializer(linear.weight)
        if use_bias is True:
            nn.init.constant_(linear.bias, bias_init)
        layers.append(linear)

        if isinstance(activation_fn, str):
            activation_fn = get_activation_fn(activation_fn)
        if activation_fn is not None:
            layers.append(activation_fn())

        self._model = nn.Sequential(*layers)

    def forward(self, x):
        return self._model(x)

class SACEncoder(nn.Module):
    def __init__(self):
        super(SACEncoder, self).__init__()
        
        self.convs = nn.Sequential(
            SlimConv2d(12, 16, kernel=[5, 5], stride=4, padding=(0, 1, 0, 1), activation_fn="relu"),
            SlimConv2d(16, 32, kernel=[5, 5], stride=2, padding=(2, 2, 2, 2), activation_fn="relu"),
            SlimConv2d(32, 32, kernel=[5, 5], stride=2, padding=(1, 2, 1, 2), activation_fn="relu"),
            SlimConv2d(32, 64, kernel=[5, 5], stride=1, padding=(2, 2, 2, 2), activation_fn="relu"),
            SlimConv2d(64, 64, kernel=[5, 5], stride=2, padding=(2, 2, 2, 2), activation_fn="relu"),
            SlimConv2d(64, 128, kernel=[5, 5], stride=2, padding=(1, 2, 1, 2), activation_fn="relu"),
            SlimConv2d(128, 256, kernel=[5, 5], stride=1, padding=0, activation_fn="relu")
        )

    def forward(self, x):
        x = x.float().permute(0, 3, 1, 2)
        conv_out = self.convs(x)
        return conv_out.squeeze(3).squeeze(2)

class SACActorNetwork(nn.Module):
    def __init__(self, encoder_output_size=256, action_dim=2):
        super(SACActorNetwork, self).__init__()
        
        self.encoder = SACEncoder()
        
        self.fc_layers = nn.Sequential(
            SlimFC(encoder_output_size, 256, activation_fn="relu"),
            SlimFC(256, 256, activation_fn="relu")
        )
        
        self.mu_head = SlimFC(256, action_dim, activation_fn=None)
        self.log_std_head = SlimFC(256, action_dim, activation_fn=None)
        
        # Action scaling
        self.action_scale = torch.tensor([0.5, 1.0])  # [steering, throttle/brake]
        self.action_bias = torch.tensor([0.0, 0.0])

    def forward(self, state, deterministic=False, with_logprob=True):
        x = self.encoder(state)
        x = self.fc_layers(x)
        
        mu = self.mu_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        
        # Pre-squash distribution and sample
        normal = Normal(mu, std)
        
        if deterministic:
            u = mu
        else:
            u = normal.rsample()
            
        action = torch.tanh(u)
        
        # Scale and shift actions
        if self.action_scale is not None:
            action = self.action_scale.to(action.device) * action
        if self.action_bias is not None:
            action = action + self.action_bias.to(action.device)

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            logp_pi = normal.log_prob(u).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - u - F.softplus(-2*u))).sum(axis=1)
        else:
            logp_pi = None

        return action, logp_pi

class SACCriticNetwork(nn.Module):
    def __init__(self, encoder_output_size=256, action_dim=2):
        super(SACCriticNetwork, self).__init__()
        
        self.encoder = SACEncoder()
        
        # Q1 architecture
        self.q1_layers = nn.Sequential(
            SlimFC(encoder_output_size + action_dim, 256, activation_fn="relu"),
            SlimFC(256, 256, activation_fn="relu"),
            SlimFC(256, 1, activation_fn=None)
        )
        
        # Q2 architecture
        self.q2_layers = nn.Sequential(
            SlimFC(encoder_output_size + action_dim, 256, activation_fn="relu"),
            SlimFC(256, 256, activation_fn="relu"),
            SlimFC(256, 1, activation_fn=None)
        )

    def forward(self, state, action):
        state_features = self.encoder(state)
        sa = torch.cat([state_features, action], 1)
        
        q1 = self.q1_layers(sa)
        q2 = self.q2_layers(sa)
        
        return q1, q2

class SACModel(nn.Module):
    def __init__(self, action_dim=2, gpu_n=0):
        super(SACModel, self).__init__()
        
        self._gpu_n = gpu_n
        
        self.actor = SACActorNetwork(action_dim=action_dim)
        self.critic = SACCriticNetwork(action_dim=action_dim)
        
        if self._gpu_n >= 0:
            self.cuda(self._gpu_n)

    def forward(self, inputs, deterministic=False):
        # Convert numpy inputs to torch tensors
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs)
        if self._gpu_n >= 0:
            inputs = inputs.cuda(self._gpu_n)
        
        # Get action from actor
        action, _ = self.actor(inputs.unsqueeze(0), deterministic=deterministic, with_logprob=False)
        
        return action.cpu().detach().numpy()[0]