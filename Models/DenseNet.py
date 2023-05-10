"""
DenseNet Model.

Involves the use of dense connections between layers, through dense blocks, where
 each layer is connected to every other layer in a feed-forward fashion. This has a 
 counter-intuitive effect of having fewer parameters than traditional conv nets, as there
 is not need to learn redundant feature maps.
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 10
GROWTH_RATE = 32

# The Dense Layer class.
class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        # Creates a layer for the dense block. Batch and relu, followed by a 1x1 conv layer and then 3x3 conv layer.
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(growth_rate * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(growth_rate * 4, growth_rate, kernel_size=3, padding=1, bias=False),
        )

    # Concatentate the output feature maps from the self.layer with the input feature maps from the previous layer.
    # Output will be tensor of dimensions (N, (C1 + (num_layers*k)), H, W), where num_layer is i from 0 to k.
    def forward(self, input):
        output = self.layer(input)
        concatentation = torch.cat([output, input], 1)
        return concatentation
    

# The Dense Block class.
class DenseBlock(nn.Module):
    def __init__(self, in_channels, num_layers, growth_rate):
        super(DenseBlock, self).__init__()
        # Calls the dense layer class to create the specified number of layers.
        self.layers = nn.Sequential(*[DenseLayer(in_channels + i * growth_rate, growth_rate) for i in range(num_layers)])

    def forward(self, input):
        return self.layers(input)
    
# The transition layers in between dense blocks.
class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionBlock, self).__init__()
        # Transistion consists of a batchnorm and relu to the input, followed by a 1x1 conv layer, then avg pooling to downsample.
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, input):
        return self.layer(input)

class DenseNet(nn.Module):
    # growth_rate is number of feature maps produced by each layer in a dense block.
    # num_layers is the number of layers for each respective dense block.
    def __init__(self, growth_rate=GROWTH_RATE, num_layers=[6,12,24,16], num_classes=NUM_CLASSES):
        super(DenseNet, self).__init__()
        in_channels = 2 * growth_rate
        # The very first convolutional layer in the network (1x1).
        self.conv1 = nn.Conv2d(3, in_channels, kernel_size=3, padding=1, bias=False)

        # The next steps is the dense blocks, followed by the transition blocks.

        # (64,32,32) -> (256,32,32) after dense block.
        self.dense1 = DenseBlock(in_channels, num_layers[0], growth_rate=growth_rate)
        in_channels += num_layers[0] * growth_rate
        out_channels = in_channels // 2
        # (256,32,32) -> (128,32,32) after transition block
        self.trans1 = TransitionBlock(in_channels, out_channels)
        in_channels = out_channels

        # (128,16,16) -> (512,16,16) after dense block.
        self.dense2 = DenseBlock(in_channels, num_layers[1], growth_rate=growth_rate)
        in_channels += num_layers[1] * growth_rate
        out_channels = in_channels // 2
        # (512,16,16) -> (256,8,8) after tranisiton block.
        self.trans2 = TransitionBlock(in_channels, out_channels)
        in_channels = out_channels

        # (258,8,8) -> (1024,8,8) after dense block.
        self.dense3 = DenseBlock(in_channels, num_layers[2], growth_rate=growth_rate)
        in_channels += num_layers[2] * growth_rate
        out_channels = in_channels // 2
        # (1024,8,8) -> (512,4,4) after transition block.
        self.trans3 = TransitionBlock(in_channels, out_channels)
        in_channels = out_channels

        # (512,4,4) -> (1024,4,4) after dense blocks.
        self.dense4 = DenseBlock(in_channels, num_layers[3], growth_rate=growth_rate)
        # no transition after, but instead simple batch, relu, then fully connnected layer.
        in_channels += num_layers[3] * growth_rate
        self.batch = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.FC = nn.Linear(in_channels, num_classes)

    def forward(self, input):
        output = self.conv1(input)
        output = self.dense1(output)
        output = self.trans1(output)
        output = self.dense2(output)
        output = self.trans2(output)
        output = self.dense3(output)
        output = self.trans3(output)
        output = self.dense4(output)
        output = self.batch(output)
        output = self.relu(output)
        # The average pooling layer makes output (1024,1,1), compatible with FC.
        output = nn.AdaptiveAvgPool2d((1,1))(output)
        # Reshape the output so that its only (1024,) shaped vector.
        output = output.view(output.size(0), -1)
        output = self.FC(output)
        return output