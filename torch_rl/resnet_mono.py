#!/usr/bin/env python3
"""DEBiT-faithful monocular encoder: half-width ResNet-18 (GroupNorm) + compression.

A port of habitat-baselines' `rl/ddppo/policy/resnet.py` (BasicBlock, ResNet,
resnet18), the backbone DEBiT's monocular branch is built on, with exactly its
structure and parameter count (`resnet18(3, 32, 16)` = 2,798,880 parameters):

    conv1 7x7 s2 p3 (no bias) -> GroupNorm -> ReLU -> MaxPool(3, 2, 1)
    -> four layers of 2 BasicBlocks at base_planes x {1, 2, 4, 8}, strides 1, 2, 2, 2

Deliberate differences from habitat-baselines, all by the amendment's spec:

  * no `avg_pool2d(2)` pre-pool: DEBiT pre-pools a 224 px sensor to 112; our
    frames are already 128 px and go in as they are;
  * no `RunningMeanAndVar`: the row stores pixels /255 in [0, 1], and the
    encoder subtracts the FIXED ImageNet mean / divides by the ImageNet std
    (constants, not buffers that update);
  * no `layer_init`: PyTorch default initialisation, which is what DEBiT's code
    effectively uses (the helper is never called). Trained from scratch.

`MonoResNetEncoder` then applies the same conv-GN-ReLU-flatten-Linear
compression as the VC-1 policy (`Vc1Compression`): at 128 px the backbone gives
(256, 4, 4), the compression conv has round(2048 / 16) = 128 channels, and the
Linear maps 2048 -> out_dim. About 4.14M parameters, DEBiT's monocular figure.

`torch_rl.gated_recurrent_tqc` is imported INSIDE `MonoResNetEncoder.__init__`
so that module can in turn import this one without a cycle.
"""
from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding, no bias."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution, no bias."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """conv3x3-GN-ReLU-conv3x3-GN, plus the residual (a conv1x1-GN downsample
    when the shape changes), then ReLU. habitat-baselines' BasicBlock."""

    expansion = 1

    def __init__(self, inplanes: int, planes: int, ngroups: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.convs = nn.Sequential(
            conv3x3(inplanes, planes, stride),
            nn.GroupNorm(ngroups, planes),
            nn.ReLU(True),
            conv3x3(planes, planes),
            nn.GroupNorm(ngroups, planes),
        )
        self.downsample = downsample
        self.relu = nn.ReLU(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.convs(x)
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class ResNetBackbone(nn.Module):
    """habitat-baselines' ResNet (BasicBlock path): (N, in_channels, H, W) ->
    (N, final_channels, H/32, W/32). No input normalisation, no pre-pool."""

    def __init__(self, in_channels: int, base_planes: int, ngroups: int,
                 block: Type[BasicBlock], layers: List[int]) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_planes, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(ngroups, base_planes),
            nn.ReLU(True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inplanes = base_planes
        self.layer1 = self._make_layer(block, ngroups, base_planes, layers[0])
        self.layer2 = self._make_layer(block, ngroups, base_planes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, ngroups, base_planes * 2 * 2, layers[2], stride=2)
        self.layer4 = self._make_layer(block, ngroups, base_planes * 2 * 2 * 2, layers[3], stride=2)

        self.final_channels = self.inplanes
        self.final_spatial_compress = 1.0 / (2 ** 5)

    def _make_layer(self, block: Type[BasicBlock], ngroups: int, planes: int, blocks: int,
                    stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.GroupNorm(ngroups, planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, ngroups, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, ngroups))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def resnet18_half(in_channels: int = 3, baseplanes: int = 32, ngroups: int = 16) -> ResNetBackbone:
    """DEBiT's `resnet18(in_channels, base_planes, ngroups)`: 2 BasicBlocks per
    layer; baseplanes 32 is the "half-width" (torchvision's ResNet-18 is 64)."""
    return ResNetBackbone(in_channels, baseplanes, ngroups, BasicBlock, [2, 2, 2, 2])


class MonoResNetEncoder(nn.Module):
    """DEBiT's monocular branch for a 128 px frame: ImageNet-normalised pixels ->
    half-width ResNet-18 (GroupNorm) -> (8*baseplanes, s/32, s/32) -> the same
    conv-GN-ReLU-flatten-Linear compression as Vc1Compression -> out_dim. No
    avg_pool2d(2) pre-pool (DEBiT pre-pools a 224 px sensor to 112; our frames
    are already 128). Trained from scratch; PyTorch default init (DEBiT's
    layer_init is never called)."""

    def __init__(self, image_size: int = 128, out_dim: int = 512, baseplanes: int = 32,
                 ngroups: int = 16, in_channels: int = 3) -> None:
        super().__init__()
        if image_size % 32:
            raise ValueError(f"image_size {image_size} must be a multiple of 32")
        if in_channels != 3:
            raise ValueError(f"in_channels must be 3 (the ImageNet mean/std are 3-channel), got {in_channels}")
        self.image_size, self.out_dim = int(image_size), int(out_dim)
        self.baseplanes, self.ngroups, self.in_channels = int(baseplanes), int(ngroups), int(in_channels)
        # Fixed constants applied AFTER the row's /255 scaling (not running statistics).
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        self.backbone = resnet18_half(self.in_channels, self.baseplanes, self.ngroups)
        g = self.image_size // 32
        self.feature_shape: Tuple[int, int, int] = (self.backbone.final_channels, g, g)
        from torch_rl.gated_recurrent_tqc import Vc1Compression   # local import: that module imports nothing from here
        self.compress = Vc1Compression(self.backbone.final_channels, grid=g, out_dim=self.out_dim)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.dim() != 4 or tuple(pixels.shape[1:]) != (self.in_channels, self.image_size, self.image_size):
            raise ValueError(
                f"expected (N, {self.in_channels}, {self.image_size}, {self.image_size}) pixels, "
                f"got {tuple(pixels.shape)}")
        # The value check is a host sync (.tolist()), which graph-breaks under Dynamo on every call, so
        # it runs in eager mode only; the shape check above is static and stays in the compiled graph.
        if not torch.compiler.is_compiling():
            lo, hi = torch.stack([pixels.detach().amin(), pixels.detach().amax()]).tolist()   # one host sync
            # Written as "not inside": a NaN anywhere makes amin/amax NaN, and every comparison
            # with NaN is False, so `hi > 1 or lo < 0` would let a NaN row through.
            if not (lo >= -1e-3 and hi <= 1.0 + 1e-3):
                raise ValueError(
                    f"pixels must be /255 in [0, 1] (the row stores them scaled), got range [{lo:.3f}, {hi:.3f}]")
        x = (pixels.float() - self.mean) / self.std
        return self.compress(self.backbone(x))

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def describe(self) -> str:
        flat = self.compress.fc.in_features
        return (f"MonoResNetEncoder half-width ResNet-18 (baseplanes {self.baseplanes}, "
                f"GroupNorm {self.ngroups}, no pre-pool, ImageNet normalisation) "
                f"{self.in_channels}x{self.image_size}x{self.image_size} -> {self.feature_shape} "
                f"-> compression {flat} -> {self.out_dim}; {self.param_count():,} params, "
                f"trained from scratch")
