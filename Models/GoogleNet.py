"""
GoogleNet Model.

Note: This specific model below do not have the auxillary classifiers included.
"""

import torch
import torch.nn as nn

NUM_CLASSES = 10

# Inception Module class. Used to create said modules, output will be concatenation of all branches.
class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionModule, self).__init__()
        # Implementation of the Inception Module. This consists of 4 branches, whose outputs will be 
        # concatenated into a single output, which will be sent to the next layer of the network.

        # branch 1: 1x1 convolution.
        self.branch1 = nn.Sequential(nn.Conv2d(in_channels, out_channels[0], kernel_size=1))
        # branch 2: 1x1 convolution, followed by 3x3 convolution.
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels[1], kernel_size=1),
            nn.Conv2d(out_channels[1], out_channels[2], kernel_size=3, padding=1)
        )
        # branch 3: 1x1 convolution, followed by 5x5 convolution.
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels[3], kernel_size=1),
            nn.Conv2d(out_channels[3], out_channels[4], kernel_size=5, padding=2)
        )
        # branch 4: 1x1 convolution, followed by max pooling.
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels[5], kernel_size=1),
        )
    
    def forward(self, input):
        # We will compute the branches in order to get outputs from each.
        branch_1 = self.branch1(input)
        branch_2 = self.branch2(input)
        branch_3 = self.branch3(input)
        branch_4 = self.branch4(input)
        # Concatenate the outputs of the branches, return it for next layer.
        concatenation = torch.cat([branch_1, branch_2, branch_3, branch_4], 1)
        return concatenation

# GoogleNet class. 
class GoogleNet(nn.Module):
    def __init__(self, num_classes = NUM_CLASSES):
        super(GoogleNet, self).__init__()
        # First layers of the network: two convolution layers (with batchnorm and max pool).
        self.convolutions = nn.Sequential(
            # Layer 1.
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Layer 2.
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        # The following layers will be the Inception Modules. Parameters sent to Inception Module class.
        # Layer 3.
        self.inceptionA3 =  InceptionModule(192, [64, 96, 128, 16, 32, 32])
        self.inceptionB3 = InceptionModule(256, [128, 128, 192, 32, 96, 64])
        # Lets end this layer by applying max pooling to the output of .inceptionB3:
        # (we can call this anytime we want. use this again after second set of inception modules.)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # Layer 4. 5 inceptions in this layer.
        self.inceptionA4 = InceptionModule(480, [192, 96, 208, 16, 48, 64])
        self.inceptionB4 = InceptionModule(512, [160, 112, 224, 24, 64, 64])
        self.inceptionC4 = InceptionModule(512, [128, 128, 256, 24, 64, 64])
        self.inceptionD4 = InceptionModule(512, [112, 144, 288, 32, 64, 64])
        self.inceptionE4 = InceptionModule(528, [256, 160, 320, 32, 128, 128])

        # Layer 5. 2 inceptions in this layer.
        self.inceptionA5 = InceptionModule(832, [256, 160, 320, 32, 128, 128])
        self.inceptionB5 = InceptionModule(832, [384, 192, 384, 48, 128, 128])
        # Lets end this layer by applying max pooling to the output of .inceptionB5.
        self.averagepool = nn.AdaptiveAvgPool2d((1,1))

        # Layer 6. Simple fully connected layer.
        self.drop = nn.Dropout(0.5)
        self.FC = nn.Linear(1024, num_classes)
    
    # Finally, lets create the forward pass of the network.
    def forward(self, input):
        # Compute initial convolution layers.
        conv_output = self.convolutions(input)

        # Now, lets compute the first set of Inception Modules.
        inception_output = self.inceptionA3(conv_output)
        inception_output = self.inceptionB3(inception_output)
        maxpool_output = self.maxpool(inception_output)

        # Now, lets compute the second set of Inception Modules.
        inception_output = self.inceptionA4(maxpool_output)
        inception_output = self.inceptionB4(inception_output)
        inception_output = self.inceptionC4(inception_output)
        inception_output = self.inceptionD4(inception_output)
        inception_output = self.inceptionE4(inception_output)
        maxpool_output = self.maxpool(inception_output)

        # Lets compute the third set of Inception Modules.
        inception_output = self.inceptionA5(maxpool_output)
        inception_output = self.inceptionB5(inception_output)
        average_output = self.averagepool(inception_output)

        # Finally, lets send this to a fully connected layer.
        # Reshape the output of the average pool so that it can be compatible with the fully connected
        # layers (since input of fully connected layers have to be flattened and not 3D tensors.)
        flattened_output = torch.flatten(average_output, 1)
        flattened_output = self.drop(flattened_output)
        x = self.FC(flattened_output)
        return x
    