from enum import Enum, auto

import torch
import torch.nn as nn


class UpsampleType(Enum):
    CONV_TRANSPOSE = auto()
    NEAREST_NEIGHBOUR = auto()
    BILINEAR = auto()


class UpConv(nn.Module):
    """
    Custom module to handle a single Upsample + Convolution block used in the decoder layer.
    Takes an optional argument stating which type of upsampling to use. This argument should be provided from the
    UpsanmpleType enum above.
    """

    def __init__(self, in_channels: int,
                 out_channels: int,
                 upsample_type: UpsampleType = UpsampleType.CONV_TRANSPOSE,
                 name=""):
        super().__init__()
        self.upsample = self._upsample(upsample_type, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same")

        self.name = name

    def _upsample(self, upsample_type: UpsampleType, num_channels: int):
        if upsample_type == UpsampleType.CONV_TRANSPOSE:
            return nn.ConvTranspose2d(num_channels,
                                      num_channels,
                                      kernel_size=2, stride=2)
        if upsample_type == UpsampleType.NEAREST_NEIGHBOUR:
            return nn.UpsamplingNearest2d(scale_factor=2)
        if upsample_type == UpsampleType.BILINEAR:
            return nn.UpsamplingBilinear2d(scale_factor=2)

        raise NotImplementedError(f"Upsampling mode of {str(upsample_type)} is not supported.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(f"Calling upsample for {self.name}")
        x = self.upsample(x)
        return self.conv(x)


class IndexedSegmentationMap(nn.Module):
    """
    Takes the raw logits from the n-channel output convolution and uses argmax to convert to an indexed output map.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.argmax(x.squeeze(), dim=0)


class ActivatedOutputConv2d(nn.Module):
    def __init__(self, input_channels: int, num_classes: int, activate: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, num_classes, kernel_size=1)
        self.activation = nn.Sigmoid() if activate else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv(x))
