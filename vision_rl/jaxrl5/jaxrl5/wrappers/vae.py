import gym
import jax.random
import numpy as np
# from wandb import gym
import cv2
from jaxrl5.networks.encoders.vae import VAE
from jaxrl5.utils.misc import load_checkpoints
import jax.numpy as jnp

class VAEWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        num_stack=6,
        stacking_key="obs",
        latent_dim=64,
        debug=False,
        observation_shape=None,
        checkpoints=None
    ):
        observation_shape=(num_stack*latent_dim,) if observation_shape is None else observation_shape
        # print(observation_shape)
        gym.Wrapper.__init__(self, env)
        self.stacking_key=stacking_key
        self.vae=VAE(latent_dim)
        if not checkpoints is None:
            self.vae_params=load_checkpoints(checkpoints,None)["params"]
        else:
            self.vae_params=self.vae.init(jax.random.PRNGKey(0), jnp.ones((1,80,160,3)))['params']
        self.observation_space = gym.spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=observation_shape,
                    dtype=np.float32,
                )


    def get_encoding(self,img):
        img = (np.array(img) / 255.0).astype(np.float32)
        img=jnp.array(img).transpose(3,0,1,2)
        img=self.vae.apply({'params':self.vae_params}, img, method=self.vae.encode)
        img=self.vae.reparameterize(*img)
        # samples=self.vae.apply({'params':self.vae_params}, img, method=self.vae.decode)
        # cv2.imwrite("sample.jpg",cv2.cvtColor((np.vstack(samples)*255).astype(np.uint8),cv2.COLOR_BGR2RGB))
        img=img.flatten()
        return img


    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        _observation=[]
        for k, v in observation.items():
             if k != "obs":
                _observation.extend(np.ravel(v).tolist())
        img=self.get_encoding(observation["obs"])
        _observation.extend(img.tolist())
        observation=np.array(_observation)
        return observation, reward, terminated,info

    def reset(self, *args, **kwargs):
        observation,info=self.env.reset(*args, **kwargs)
        # print(observation.shape)
        _observation=[]
        for k, v in observation.items():
             if k != "obs":
                _observation.extend(np.ravel(v).tolist())
        img=self.get_encoding(observation["obs"])
        _observation.extend(img.tolist())
        observation=np.array(_observation)
        # print(info)
        return observation


