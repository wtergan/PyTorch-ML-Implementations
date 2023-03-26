# PyTorch ML Implementations

Common Architectures used for classification on the CIFAR-10 Dataset.

The CIFAR-10 dataset is a well-known benchmark dataset in the field of computer vision and machine learning. It consists of 60,000 32x32 color images that are divided into 10 classes, with each class containing 6,000 images. The dataset is split into a training set and a test set, with 50,000 and 10,000 images respectively.

The classes in the CIFAR-10 dataset include airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. These images were collected from the internet and hand-labeled to ensure accurate classification.

The CIFAR-10 dataset served as a benchmark for evaluating the performance of various machine learning algorithms, particularly in the area of image classification. 

Below is the desciptions of the various model architectures I have implemented using Pytorch as the framework, but first, lets show the default parameters I've used.

### Parameters (default, can be changed):

        --lr                default=1e-3    Learning Rate.
        --epoch             default=50      Number of epochs in the training loop.
        --trainbatchsize    default=100     Size of the train batch.
        --testbatchsize     default=100     Size of the test batch.

Now, lets decribe the various model architectures thats been implemented:

### AlexNet:

Paper : https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf

"ImageNet Classification with Deep Convolutional Neural Networks
    Alex Krizhevsky, Ilya Stuskever, Geoffery E. Hinton.

In the traditional paper, the model consisted of:

    - Five convolutional layers and three fully connected layers.
    
    - Trained on ImageNet, where each image in the dataset is a 224x224x3 image.
    
    - Data Argmentation implmented on the images included:
    
        - Random cropping of the images.
        
        - Random horizontal flips of the images (probability was 50%).
        
        - Color normalization (subtracting the mean activity over the training set from each pixel).
        
    - Since we are going to train the model on the CIFAR-10 dataset which consists of 32x32x3 images, we will have to slightly modify the model to handle these images.

In this modified version of AlexNet:

    - We will train the model on CIFAR-10, not ImageNet, so the images put into the model will be 32x32x3, not 224x224x3.
    
    - We will not implement Local Response Normalizaiton to the convolutional layers, which was originally in the paper. It is not commonly used anymore. Instead, we will apply 2D Batchnorm to each of the layers.
    
    - For data augementation, we will only:
    
        - Randomly crop the images.
        
        - Randomly flip the images horizontall;y, and instead of the probabilty being 50%, it will instead be all of the images.
        
        - We will not implement color normalization on the training set.

#### VGGNet:

Paper: https://arxiv.org/pdf/1409.1556.pdf

" Very Deep Convolutional Networks For Large-Scale Image Recognition"
    Karen Simonytan, Andrew Zisserman

In the traditional paper, Karen and Andrew demonstrated 5 different model architectures, being named A-E:

    - A: 11 Layers, 8 of which being convolutional layers.
    
    - A-LRN: 11 Layers, with Local Response Normalization included in the first convolution layer.
    
    - B: 13 Layers, 10 of which being convolutional layers.
    
    - C: 16 Layers, 13 of which being convolutional layers.
    
    - D: 16 Layers, 13 of which being convolutional layers, the difference from C being that they modified sized of some of the layers.
    
    - E: 19 Layers, 16 of which being convolutional layers.

    - In each of the model archtectures, they incorporated 1 x 1 convolutional layers, as a way to increase the non-linearity of the decision function without affecting the receptive fields of the convoltional layers.

    - While it did not win the ImageNet challenge, it was 2nd place in 2014 (GoogleNet won in 2014).

    - Data augmentation implemented on the images included:
        - Random scaling and then cropping of the images.
        - Random horizontal flipping of the images.
        - Color jittering of the images, which means that the RBG color channels of the images were altered by adding a random value drawn vrom a normal distribution with zero mean and 0.1 standard deviation.
        - Also, the paper mentioned the technique of using multi-crop evaluation at test time, meaning that they took multiple 224x224 crops from the test image adn averageing the predictions made by the network on these aformentioned crops.

In this modified version of VGGNet, we will:

    - Only implement the smallest of the demonstrated model architecture, that being the model A, which consists of 11 layers.
    
    - For data augmentation, we will:
    
        - Randomly crop the images, we will not implement scaling of the images.
        
        - Randomly flip the images horizontally, flipping all of the images in the training set.
        
        - We will not implement multi-crop evalutatons of the images at test time.

#### GoogleNet

Paper: https://arxiv.org/abs/1409.4842v1

"Going Deeper With Convolutions"
    Christen Szegedy, Wei Liu, Yangqing Jia, et al.

In this paper, the authors developed a new neural network arcitecture called the 'Inception Module", which improved the utilization of computing resources inside of said network. This novel idea increased the performance of the model on the ImageNet dataset, and won the ILSVRC in 2014.

The Inception Module was inspired by the "Network in Network" paper, also published in 2014 (https://arxiv.org/pdf/1312.4400.pdf). This paper described, 'building micro neural netowrks with more complex structues to abstract the data with in the receptive field' (page 1). This was to be an improvement to simply using linear filters followed by non-linear functions. 

In the paper by Szegedy et al., the original inception module consists of four parallel branches:

    - 1x1 convolution, 3x3 convolution, 5x5 convolution, and a 3x3 max pool.
    
    - Each of the branches had a different number of filters, and these computations were followed by a batch norm and a ReLU.
    
    - Outputs of these branches were concatenated along the channel dimension.
    
    - That concatenated output will then be sent to the next layer.
    
    - They improved the modules by placing 1x1 convolutions before the 3x3 and 5x5 to reduce the dimensionality of the input.
    
    - They also replaced the 5x5 convolution with two consecutive 3x3 convolutions, which will produce the same output, but will involve fewer parameters, thus making it more efficient.

An important thing to know that has been outlined in the paper is that 'as features of higher abstraction are captured by higher layers, their spatial concentration is expected to decrease, suggesting that the ratio of 3x3 and 5x5 convolutions should increase as we move to higher layers' (page 4).

Since the model is way more complicated (convoluted haha) than other models that came before it, lets outline each of the model in its entirety (this will be our modified version for simplicity):

    - Layer 1:
    
        - Convolution: (conv - batch - relu - maxpool)
        
    - Layer 2:
    
        - Convolution:  (conv - batch - relu - maxpool)

    - Layer 3:
    
        - Inception (layer 3a):
        
            - Branch 1: 1x1 convolution.
            
            - Branch 2: 3x3 convolution, with 1x1 dimensionality reduction.
            
            - Branch 3: 5x5 convolution, with 1x1 dimensionaity reduction.
            
            - Branch 4: 3x3 maxpool, with 1x1 dimensionality reduction. 
            
            - Concatenation of all of the branches' outputs.
            
        - Inception (layer 3b):
        
        - Max Pool for concatenated output.

    - Layer 4:
    
        - Inception (layer 4a):
        
        - Inception (layer 4b):
        
        - Inception (layer 4c):
        
        - Inception (layer 4d):
        
        - Inception (layer 4e):
    
    - Layer 5:
    
        - Inception (layer 5a):
        
        - Inception (layer 5b):
        
        - Average Pool for concatenated output.

    - Layer 6:
    
        - Dropout.
        
        - Fully connected.

### Performance Measure:

I've ran these models on a M1 Macbook Air, with 256GB of storage and 7GPUs. Since I'm running these models on this laptop, I've only trained on 50 epochs. Thus, the performance measure below is based on my hardware and limited number of epochs (should run for at least 100 epoch or more for even better performance).

AlexNet: Training Accuracy: 82.65%  Testing Accuracy: 80.75%

VGG (11 layer variant):

GoogLeNet:

ResNet18:

DenseNet121:

DenseNet169:
