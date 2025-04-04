# model_def.py
"""
Defines the ResNet-like model with dynamic pooling capabilities.
Includes block definitions and helper functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

# === Pooling Function ===
def pooling_func(x: torch.Tensor) -> torch.Tensor:
    """Applies Max Pooling to halve spatial dimensions."""
    return F.max_pool2d(x, kernel_size=2)

# === Convolution Helpers ===
def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Module:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Module:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# === Block Definitions ===
class ResBasicBlock(nn.Module):
    """Standard ResNet Basic Block."""
    expansion = 1
    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        # Using non-standard BN params
        bn_kwargs = {'affine': False, 'track_running_stats': False}
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, **bn_kwargs)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, **bn_kwargs)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out)
        if self.downsample is not None: identity = self.downsample(x)
        out += identity # Skip connection
        out = self.relu(out)
        return out

class BasicConvBlock(nn.Module):
    """Simple Conv-BN-ReLU Block."""
    def __init__(self, in_planes: int, out_planes: int, kernel_size: int = 3):
        super().__init__()
        bn_kwargs = {'affine': False, 'track_running_stats': False}
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, **bn_kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x); out = self.bn(out); out = self.relu(out)
        return out

# === Dynamic ResNet Model ===
class ResNet20(nn.Module):
    """
    A ResNet-like architecture where pooling before stages is dynamically controlled.
    """
    def __init__(self, block: type, layers_in_stages_unused: Optional[List[int]],
                 channels: List[int], num_classes: int = 10):
        super().__init__()
        assert len(channels) > 0, "Channels list cannot be empty"
        self.num_stages = len(channels)
        self.inplanes = channels[0]
        self._current_path: Optional[Tuple] = None

        self.stage0 = BasicConvBlock(3, self.inplanes, kernel_size=3)
        stages = []
        for i in range(1, self.num_stages):
            stages.append(self._make_layer(block, channels[i]))
        self.stages = nn.ModuleList(stages)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.inplanes, num_classes)
        print(f"      [Model INFO] Created ResNet. Stages: {self.num_stages}. Final FC features: {self.inplanes}")

    def set_path(self, path: Tuple):
        """Sets the pooling configuration path for the next forward pass."""
        if path is None or len(path) != self.num_stages:
             raise ValueError(f"Path length {len(path) if path else 'None'} != stages {self.num_stages}")
        self._current_path = path

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dynamic pooling based on the set path."""
        if self._current_path is None: raise RuntimeError("Path not set via set_path()")
        path = self._current_path
        if path[0]: x = pooling_func(x)
        x = self.stage0(x)
        for i in range(len(self.stages)):
            stage_index = i + 1
            if path[stage_index]: x = pooling_func(x)
            x = self.stages[i](x)
        x = self.avgpool(x); x = torch.flatten(x, 1); x = self.fc(x); return x

    def _make_layer(self, block: type, planes: int, blocks: int = 1, stride: int = 1) -> nn.Module:
        """Helper function to create a ResNet stage."""
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, affine=False, track_running_stats=False),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

# === Model Creation Helper ===
def create_model(num_stages: int, channels: List[int]) -> nn.Module:
    """Creates a single ResNet20 instance based on channels list."""
    print(f"      Creating model with {num_stages} stages, channels: {channels}")
    return ResNet20(ResBasicBlock, layers_in_stages_unused=None, channels=channels)