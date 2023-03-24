# PyTorch ML Implementations

Common Architectures used for classification on the CIFAR-10 Dataset.

The CIFAR-10 dataset is a well-known benchmark dataset in the field of computer vision and machine learning. It consists of 60,000 32x32 color images that are divided into 10 classes, with each class containing 6,000 images. The dataset is split into a training set and a test set, with 50,000 and 10,000 images respectively.

The classes in the CIFAR-10 dataset include airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. These images were collected from the internet and hand-labeled to ensure accurate classification.

The CIFAR-10 dataset served as a benchmark for evaluating the performance of various machine learning algorithms, particularly in the area of image classification. 

The model we are going to make is a slight modification of AlexNet.

Paper : https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf

"ImageNet Classification with Deep Convolutional Neural Networks
    Alex Krizhevsky, Ilya Stuskever, Geoffery E. Hinton.

In the traditional paper, the model consisted of:
        - Five convolutional layers and three fully connect layers.
        - Trained on ImageNet, where each image in the dataset is a 224x224x3 image, not CIFAR-10.
        - since we are going to train the model on the CIFAR-10 dataset which consists of 32x32x3 images, we will have to slightly modify the model to handle these images.

Parameters (default, can be changed):

--lr                default=1e-3    Learning Rate.
--epoch             default=50      Number of epochs in the training loop.
--trainbatchsize    default=100     Size of the train batch.
--testbatchsize     default=100     Size of the test batch.

AlexNet Architecture:

   (CONV-BATCH-RELU-MAXPOOL) x 2 - (COV-BATCH-RELU) x 2 - (CONV-BATCH-RELU-MAXPOOL) - (DROP-FC-BATCH--RELU) x 2 - (FC-SOFTMAX)

        - We will not implement Local Response Normalization to the convolutional layers, which was originally in the paper. It is not commonly used anymore. Instead, we will apply 2D Batchnorm to each of the layers.

        - Data augmentation involved randomly cropping annd horizontally flipping the images in the training set.

Performance Measure:

AlexNet: Training Accuracy: 89%  Testing Accuracy: 76.6%
VGG11:
VGG16:
VGG19:
GoogLeNet:
ResNet18:
DenseNet121:
DenseNet169:
