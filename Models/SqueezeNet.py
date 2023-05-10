"""
SqueezeNet Model.

Achieves high accuracy with fewer parameters (when compared to AlexNet).

Note: This is the simple 1.0 version of the mode; (do not have residual blocks).
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 10

# The Fire module. Consists of a squeeze conv layer (1x1 filters), followed by an expand conv layer
# (1x1 and 3x3 filters). Squeeze layer should be less than the summation of filters in the expand layer.
class FireModule(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand_channels):
        super(FireModule, self).__init__()
        # Squeeze layer.
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channels, squeeze_channels, kernel_size=1),
            nn.BatchNorm2d(squeeze_channels),
            nn.ReLU(inplace=True)
        )
        # Expand layers: one 1x1 conv followed by one 3x3 conv.
        # Expand 1.
        self.expand_1x1 = nn.Sequential(
            nn.Conv2d(squeeze_channels, expand_channels, kernel_size=1),
            nn.BatchNorm2d(expand_channels),
            nn.ReLU(inplace=True)
        )
        # Expand 2.
        self.expand_3x3 = nn.Sequential(
            nn.Conv2d(squeeze_channels, expand_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(expand_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Squeeze layer computation.
        x_squeezed = self.squeeze(x)
        # Expand layer (which itself consists of Expand 1 and 2).
        x_expanded_1x1 = self.expand_1x1(x_squeezed)
        x_expanded_3x3 = self.expand_3x3(x_squeezed)
        # Concatenate the two aforementioned expand layers and send the result to ReLU for non-linearity.
        return torch.cat([x_expanded_1x1, x_expanded_3x3], 1)


class SqueezeNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(SqueezeNet, self).__init__()
        # The initial convolutional layer.
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Firing of the fire modules.
            FireModule(96, 16, 64),
            FireModule(128, 16, 64),
            FireModule(128, 32, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            FireModule(256, 32, 128),
            FireModule(256, 48, 192),
            FireModule(384, 48, 192),
            FireModule(384, 64, 256)
        )
        # Final convolutional layer, do not need fully connected afterwards.
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, input):
        output = self.features(input)
        output = self.classifier(output)
        # Reshaping of the output from the .classifer.
        return output.view(output.size(0), -1)


    




              
