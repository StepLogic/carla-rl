import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence

class ConvBlock(nn.Module):
    features: int
    kernel_size: int = 4
    strides: int = 2

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=self.features, kernel_size=(self.kernel_size, self.kernel_size),
                    strides=(self.strides, self.strides), padding='VALID')(x)
        x = nn.relu(x)
        return x

class ConvTransposeBlock(nn.Module):
    features: int
    kernel_size: int = 4
    strides: int = 2

    @nn.compact
    def __call__(self, x):
        x = nn.ConvTranspose(features=self.features, kernel_size=(self.kernel_size, self.kernel_size),
                             strides=(self.strides, self.strides), padding='VALID')(x)
        x = nn.relu(x)
        return x

class VAE(nn.Module):
    latent_size: int = 64

    def setup(self):
        self.image_size = (80, 160, 3)
        self.encoder = [
            ConvBlock(features=32),
            ConvBlock(features=64),
            ConvBlock(features=128),
            ConvBlock(features=256),
        ]
        self.encoded_H, self.encoded_W, _ = self._calculate_spatial_size(self.image_size, [4, 4, 4, 4], [2, 2, 2, 2])
        self.flatten_size = self.encoded_H * self.encoded_W * 256

        self.mean = nn.Dense(features=self.latent_size)
        self.logstd = nn.Dense(features=self.latent_size)
        self.latent = nn.Dense(features=self.flatten_size)

        self.decoder = [
            ConvTransposeBlock(features=128),
            ConvTransposeBlock(features=64),
            ConvTransposeBlock(features=32, kernel_size=5),
            ConvTransposeBlock(features=3),
        ]

    def encode(self, x):
        for layer in self.encoder:
            x = layer(x)
        x = x.reshape((x.shape[0], -1))
        return self.mean(x), self.logstd(x)

    def decode(self, z):
        z = self.latent(z)
        z = z.reshape((-1, self.encoded_H, self.encoded_W, 256))
        for layer in self.decoder[:-1]:
            z = layer(z)
        z = self.decoder[-1](z)
        return nn.sigmoid(z)

    def reparameterize(self, mu, logvar):
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(jax.random.PRNGKey(0), logvar.shape)
        return mu + eps * std

    def __call__(self, x, encode=False, mean=False):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x = self.decode(z)
        return x, mu, logvar

    def _calculate_spatial_size(self, image_size, kernel_sizes, strides):
        H, W, _ = image_size
        for kernel_size, stride in zip(kernel_sizes, strides):
            H = (H - kernel_size) // stride + 1
            W = (W - kernel_size) // stride + 1
        return H, W, 256  # 256 is the number of features in the last conv layer
