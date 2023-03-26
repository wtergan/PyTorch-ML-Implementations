"""
AlexNet Model.
"""

import torch.nn as nn

NUM_CLASSES = 10

class AlexNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(AlexNet, self).__init__()
        # Implementation of the first five convolutional layers, which will extract various high level 
        # features of each image.
        # We will also apply inplace=True in RELU so as to perform RELU on input directly. This saves 
        # memory and computational time.
        self.convolutions = nn.Sequential(
            # Layer 1. 
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # Layer 2.
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # Layer 3.
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Layer 4.
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Layer 5.
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        
        # Lets now implement the last three layers of the network, which is the fully connect layers.
        self.fullyconnected = nn.Sequential(
            # Layer 6.
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            # Layer 7.
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # Layer 8.
            # Now that we do not have to include softmax, as nn.CrossEntropyLoss handles this for us.
            nn.Linear(4096, num_classes),
        )

    # Finally, lets create the forward pass of the network.
    def forward(self, input):
        input = self.convolutions(input)
        # Reshape the output of the last convolution layer so that it can be compatible with the fully connected
        # layers (since input of fully connected layers have to be flattened and not 3D tensors.)
        input = input.view(input.size(0), 256 * 2 * 2)
        input = self.fullyconnected(input)
        return input
