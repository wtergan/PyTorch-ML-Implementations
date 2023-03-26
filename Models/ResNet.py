"""
ResNet Model.

Note: The version of ResNet demonstrated below is the 18-layer variant.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 10

class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # Implementation of the residual blocks, used to learn identity mappings with reference to layer's inputs.
        # Below is the initialization of the first convolutional layer in the block. 
        # Note, bias is set False. This is because we do not need it for learning identity mappings.
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.batch1 = nn.BatchNorm2d(out_channels)
        
        # Initialization of the second layer of the block.
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection creation. y = F(x, {Wi}) + Ws*x, where Ws*x is the shortcut connection.
        # In the line below, we initialize an empty sequential container. If the input to this shortcut
        # and the Ws (weight matrix) is the same dimension, then Ws is the "identity matrix" for 
        # shortcut and the shortcut connection will be an "identity mapping".
        self.shortcut = nn.Sequential()
        # If the dimensions is not the same, then Ws (weight matrix) is NOT the same, thiu we will have to make the
        # dimensions the same by applying a 1x1 conv and batch. Result is a "linear projection mapping".
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, input):
        # First layer of the residual function. Consists of the 1st convolution, batchnorm, and then RELU activation function.
        out = self.conv1(input)
        out = self.batch1(out)
        out = F.relu(out)
        
        # Second layer of the residual function. Consists of the second convolution, and then batchnorm.
        out = self.conv2(out)
        out = self.batch2(out)
        
        # Add the shortcut connection to the input. the result is a "residual block".
        # y = F(x, {Wi}) + Wsx, where F(x, {Wi}) = the residual function.
        # Output is either an identity mapping (if Ws is identity matrix), or lienar projection mapping (is Ws NOT identity matrix).
        out += self.shortcut(input)
        
        # Now, we can apply RELU non-linearity to the output of the residual block.
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=NUM_CLASSES):
        super(ResNet, self).__init__()
        self.in_channels = 64

        # The very first convolutional layer to start the model.
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch1 = nn.BatchNorm2d(64)
        
        # We will create the layers of Residual blocks. 
        self.block1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.block2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.block3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.block4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # Output fully connected layer for classification.
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    # This function is used to create the layers of the Residual blocks. 
    # Takes in block (ResidualBlock class), number of output channels for each block, the num layers for the block,
    # and the stride.
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            # Add the Residual blocks to the layer list.
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
            # This will compute the layers that constitute the Residual block. 
        return nn.Sequential(*layers)

    def forward(self, input):
        # Pass through initial convolutional layer and activation.
        out = self.conv1(input)
        out = self.batch1(out)
        out = F.relu(out)
        
        # Pass through the layers with basic blocks.
        out = self.block1(out) # 3 layers.
        out = self.block2(out) # 5 layers.
        out = self.block3(out) # 6 layers.
        out = self.block4(out) # 3 layers.
        
 # Average pooling layer with kernel size 4x4 to reduce spatial dimensions
        out = F.avg_pool2d(out, 4)
        
        # Flatten the output to prepare for the fully connected layer.
        out = out.view(out.size(0), -1)
        
        # Fully connected layer.
        out = self.linear(out)
        return out

# Function to create a ResNet-34 model
def resnet34():
    return ResNet(ResidualBlock, [3, 4, 6, 3])
