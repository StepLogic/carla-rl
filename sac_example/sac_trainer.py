#!/usr/bin/env python

import os
import torch
from ray.rllib.agents.dqn import SACTrainer

class CustomDQNTrainer(SACTrainer):
    def setup(self, config):
        SACTrainer.setup(self, config)  # Try direct parent class call instead of super()
    
    def save_checkpoint(self, checkpoint_dir):
        checkpoint_path = SACTrainer.save_checkpoint(self, checkpoint_dir)
        model = self.get_policy().model
        torch.save(
            model.state_dict(),
            os.path.join(checkpoint_dir, "checkpoint_state_dict.pth")
        )
        return checkpoint_path