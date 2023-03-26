"""
VGG Net Model.
"""

import torch.nn as nn

NUM_CLASSES = 10

class VGGNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(VGGNet, self).__init__()

        # Implementation of the first eiight convolutional layers, which will extract various high level 
        # features of each image.

        # We will also apply inplace=True in RELU so as to perform RELU on input directly. This saves 
        # memory and computational time.

        # Max pooling used downsample the activation maps (output) of some layers.
        self.convolutions = nn.Sequential(
            # Layer 1.
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Layer 2.
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Layer 3.
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Layer 4.
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Layer 5.
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Layer 6.
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Layer 7.
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Layer 8.
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Lets now implement the last three layers of the network, which is the fully connected layers.       
        self.fullyconnected = nn.Sequential(
            # Layer 9.
            nn.Dropout(),
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(inplace=True),
            # Layer 10.
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # Now that we do not have to include softmax, as nn.CrossEntropyLoss handles this for us.
            # Layer 11.
            nn.Linear(4096, num_classes),
        )

    
    # Finally, lets create the forward pass of the network.
    def forward(self, input):
        input = self.convolutions(input)
        # Reshape the output of the last convolution layer so that it can be compatible with the fully connected
        # layers (since input of fully connected layers have to be flattened and not 3D tensors.)
        input = input.view(input.size(0), 512 * 1 * 1)
        input = self.fullyconnected(input)
        return input
    