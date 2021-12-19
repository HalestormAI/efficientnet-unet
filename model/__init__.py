from typing import Dict

import torch
import torch.nn as nn
from torchvision.models.efficientnet import (
    efficientnet_b0,
    efficientnet_b1,
    efficientnet_b2,
    efficientnet_b3,
    efficientnet_b4,
    efficientnet_b5,
    efficientnet_b6,
    efficientnet_b7,
    MBConv
)

MODELS = [
    efficientnet_b0,
    efficientnet_b1,
    efficientnet_b2,
    efficientnet_b3,
    efficientnet_b4,
    efficientnet_b5,
    efficientnet_b6,
    efficientnet_b7
]

BlockName = str

from model.blocks import (
    UpConv, UpsampleType, ActivatedOutputConv2d
)


class EffUnet(nn.Module):
    """
    Asymmetrical Unet-style model using EfficientNet as an encoder. Based loosely on this paper:

    Baheti, B., Innani, S., Gajre, S.S., & Talbar, S.N. (2020).
    Eff-UNet: A Novel Architecture for Semantic Segmentation in Unstructured Environment. 2020
    IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 1473-1481.
    https://openaccess.thecvf.com/content_CVPRW_2020/papers/w22/Baheti_Eff-UNet_A_Novel_Architecture_for_Semantic_Segmentation_in_Unstructured_Environment_CVPRW_2020_paper.pdf
    """

    def __init__(self,
                 model_size: int = 0,
                 num_classes: int = 2,
                 upsample_type: UpsampleType = UpsampleType.CONV_TRANSPOSE,
                 remove_bn: bool = False,
                 activate_logits = True):
        super().__init__()
        norm_type = nn.Identity if remove_bn else None

        self.upsample_type = upsample_type

        # The paper implies a set of skip connections that don't actually work in reality due to repeated blocks
        # of the same scale clashing with the upsamples. The paper suggests (2,3,4,6,7), but for dimensions to agree
        # it needs to be as below.
        self.skipped_blocks = (1, 2, 3, 5, 7)

        # Only need the stem, don't want to use the classifier or avg pooling at the tail of the effnet as part of this
        # model
        effnet = MODELS[model_size](pretrained=not remove_bn, norm_type=norm_type)
        self.encoder = effnet.features[:-1]

        self.block_channels = self._get_skip_channels()

        # This is required to extract the skip connections from the model - note that this won't run until the fwd
        # pass is actually running, which is a bit grim.
        self.skip_connections = {}
        self._register_skip_hooks()

        self.upsample_channels = [16, 64, 128, 256, 512]
        self.upsample_ops = self._generate_upsample_ops()

        self.cls_conv = ActivatedOutputConv2d(self.upsample_channels[0], num_classes, activate=activate_logits)

    def _get_skip_channels(self) -> Dict[BlockName, int]:
        """
        Run through the effnet model and extract the number of channels coming out of each effnet block so that we can
        build the ops for the decoder.
        :return: dictionary of block names against the number of channels output by the block
        """
        # First iterate through the encoder layers from first to last and capture the number of output channels from
        # each MBConv. By filtering like this, we ignore the first layer of effnet (just a conv).
        bc = [f[-1].out_channels for f in self.encoder if isinstance(f[-1], MBConv)]

        # Now build the dictionary against block names. bc is 0-indexed, but the block names are 1-indexed.
        return {f"block_{i + 1}": b for i, b in enumerate(bc) if i + 1 in self.skipped_blocks}

    def _generate_upsample_ops(self) -> nn.ModuleDict:
        """
        Create the operators for the decoder. Although we can't actually build this at construction time, because the
        skip connections have to be extracted using a fwd hook, we can at least build the ops here. We keep track of
        the number of channels in the previous decoder layer to make sure the input sizes are correct for the upsample.
        TODO: Make upsample method switchable between convtranspose, NN and bilinear upsample.
        :return:
        """
        layers = {}
        out_channels = 0
        channels = [x for x in self.upsample_channels]
        for i in self.skipped_blocks[::-1]:
            block_name = f"block_{i}"
            in_channels = self.block_channels[block_name] + out_channels
            out_channels = channels.pop()
            layers[block_name] = UpConv(in_channels, out_channels, self.upsample_type, name=block_name)
        return nn.ModuleDict(layers)

    def _register_skip_hooks(self) -> None:
        """
        Register forward hooks for all skip connections from encoder => decoder so that when we pass through the model
        in the fwd pass, we can connect the encoder outputs to the decoder layers.
        :return:
        """

        def get_activation(name):
            def hook(model: nn.Module, input_act: torch.Tensor, output_act: torch.Tensor):
                self.skip_connections[name] = output_act

            return hook

        for i, block in enumerate(self.encoder):
            # Skip the first element as it's just a straight conv. Since the paper lists the blocks using 1-indexing,
            # the numbers still tally up.
            if i in self.skipped_blocks:
                block.register_forward_hook(get_activation(f'block_{i}'))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)

        for i in self.skipped_blocks[::-1]:
            block_name = f"block_{i}"
            skip = self.skip_connections[block_name]
            upconv = self.upsample_ops[block_name]

            if i == self.skipped_blocks[-1]:
                x = skip
            else:
                x = torch.cat([x, skip], dim=1)
            x = upconv(x)

        logits = self.cls_conv(x)
        return logits
