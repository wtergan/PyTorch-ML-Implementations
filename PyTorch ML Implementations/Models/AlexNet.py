"""
AlexNet Model.

    Paper: 'ImageNet Classification with Deep Convolutional Neural Networks'
        : https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf

Model Architecture.

    - The model consists of eight layers: five convolutional layers and three fully connect layers.
    - The model was trained on ImageNet, where each image in the dataset is a 224x224x3 image.
        since we are going to train the model on the CIFAR-10 dataset which consists of 32x32x3 images, we
        will have to slightly modify the model to handle these images.

    (CONV-BATCH-RELU-MAXPOOL) x 2 - (COV-BATCH-RELU) x 2 - (CONV-BATCH-RELU-MAXPOOL) - (DROP-FC-BATCH--RELU) x 2 - (FC-SOFTMAX)

    - We will not implement Local Response Normalization to the convolutional layers, which was originally in the 
       paper. It is not commonly used anymore. Instead, we will apply 2D Batchnorm to each of the layers.

    - Data augmentation involved randomly cropping annd horizontally flipping the images in the training set.
"""

import torch.nn as nn

NUM_CLASSES = 10

class AlexNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(AlexNet, self).__init__()
        # implementation of the first five convolutional layers, which will extract various high level 
        # features of each image.
        # We will also apply inplace=True in RELU so as to perform RELU on input directly. This saves 
        # memory and computational time.
        self.convolutions = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        
        # Lets not implement the last three layers of the network, which is the fully conenct layers.
        self.fullyconnected = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # Not that we do not have to include softmax, as nn.CrossEntropyLoss handles this for us.
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
