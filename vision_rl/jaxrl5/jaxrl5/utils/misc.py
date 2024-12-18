# th.autograd.set_detect_anomaly(True)
# import torch as th
import ssl
import os

from PIL import Image
import torch
from torch.utils.data import Dataset



import numpy as np
from jax.tree_util import tree_map
from torch.utils import data

# th.autograd.set_detect_anomaly(True)
ssl._create_default_https_context = ssl._create_stdlib_context
import os
from flax.core.frozen_dict import unfreeze
from flax.training import checkpoints
import numpy as np
def load_pretrained(checkpoint_dir,agent, obs_space=None, ac_space=None):
    loaded_params = checkpoints.restore_checkpoint(checkpoint_dir, target=None)

    actor_params = unfreeze(agent.actor.params)
    for k, v in loaded_params["actor"]["params"].items():
        if 'encoder' in k: #load only encoder
            actor_params[k] = v
    critic_params = unfreeze(agent.critic.params)

    return agent.replace(
        actor=agent.actor.replace(
            params=unfreeze(actor_params),
        ),
        critic=agent.critic.replace(
            params=unfreeze(critic_params),
        ),
        target_critic=agent.critic.replace(
            params=unfreeze(critic_params),
        ),
    )

def load_checkpoints(checkpoint_dir,agent, obs_space=None, ac_space=None):
    loaded_params = checkpoints.restore_checkpoint(checkpoint_dir, target=agent)
    return loaded_params


def swap_axis(pic):
    img = pic.permute((1, 2, 0)).contiguous()
    # put it from HWC to CHW format
    return img

##########################################
class ImageDataset(Dataset):
    def __init__(self, folder,transform=None,y_transform=None,crop_y=False, image_size=(32,32)):
        self.folder = folder
        self.image_names = os.listdir(folder)
        self.img_size=image_size
        self.crop_y=crop_y
        from torchvision.transforms import v2
        ##################################################
        self.base = v2.Compose([
                v2.Resize(image_size),
                v2.ToTensor()  # p is the probability of applying the transformation
            ])
        self.transform = transform if transform else lambda x: x
        self.y_transform = y_transform if y_transform else lambda x: x
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
    
        image_name = self.image_names[idx]
        x = Image.open(os.path.join(self.folder, image_name)).resize(self.img_size)
        if self.crop_y:  
             y = Image.open(os.path.join(self.folder, image_name)).crop((0,42,84,84)).resize(self.img_size)
        else:
             y = Image.open(os.path.join(self.folder, image_name)).resize(self.img_size)
        return self.transform(self.base(x)),self.y_transform( self.transform(self.base(y)))


class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform if transform else lambda x: x
    
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        # return torch.stack([ swap_axis(x) for _ in range(6)],dim=-1),  torch.stack([ swap_axis(y) for _ in range(6)],dim=-1)
        return swap_axis(x),swap_axis(y)
    def __len__(self):
        return len(self.subset)

######################################
def numpy_collate(batch):
  return tree_map(np.asarray, data.default_collate(batch))

class JaxDataLoader(data.DataLoader):
  def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
    
    super(self.__class__, self).__init__(dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=numpy_collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn)