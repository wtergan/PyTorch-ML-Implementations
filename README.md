# PyTorch ML Implementations

Common Architectures used for classification on the CIFAR-10 Dataset.

The CIFAR-10 dataset is a well-known benchmark dataset in the field of computer vision and machine learning. It consists of 60,000 32x32 color images that are divided into 10 classes, with each class containing 6,000 images. The dataset is split into a training set and a test set, with 50,000 and 10,000 images respectively.

The classes in the CIFAR-10 dataset include airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. These images were collected from the internet and hand-labeled to ensure accurate classification.

Lets look at some of the images in the dataset so that we can have an understanding of what we are training the model on.

Below is a grid of the first 7 images in each class (airplace, automobile, bird, etc.):


Normal Image:

   <img src="https://github.com/wtergan/PyTorch-ML-Implmentations/blob/main/Data%20Images/Normal_images.png" width=80% height=80%>

Flipped Image:

   <img src="https://github.com/wtergan/PyTorch-ML-Implmentations/blob/main/Data%20Images/Flipped_imagespng.png" width=80% height=80%> 

Cropped Image:

   <img src="https://github.com/wtergan/PyTorch-ML-Implmentations/blob/main/Data%20Images/Cropped_images.png" width=80% height=80%> 

Color-Jittered Image:

   <img src="https://github.com/wtergan/PyTorch-ML-Implmentations/blob/main/Data%20Images/Color_jittered.png" width=80% height=80%> 


The CIFAR-10 dataset served as a benchmark for evaluating the performance of various machine learning algorithms, particularly in the area of image classification. 

Below is the desciptions of the various model architectures I have implemented using Pytorch as the framework, but first, lets show the default parameters I've used.

### Parameters (default, can be changed):

- `--lr` default=1e-3: Learning Rate.
- `--epoch` default=50: Number of epochs in the training loop.
- `--trainbatchsize` default=100: Size of the train batch.
- `--testbatchsize` default=100: Size of the test batch.

Now, lets decribe the various model architectures thats been implemented:

### AlexNet:

Paper : https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf

**"ImageNet Classification with Deep Convolutional Neural Networks"**

   ---*Alex Krizhevsky, Ilya Stuskever, Geoffery E. Hinton.*

In the traditional paper, the model consisted of:

- Five convolutional layers and three fully connected layers.
    
- Trained on ImageNet, where each image in the dataset is a 224x224x3 image.
    
- Data Argmentation implmented on the images included:
    
    - Random cropping of the images.
        
    - Random horizontal flips of the images (probability was 50%).
        
    - Color normalization (subtracting the mean activity over the training set from each pixel).
        
Since we are going to train the model on the CIFAR-10 dataset which consists of 32x32x3 images, we will have to slightly modify the model to handle these images.

In this modified version of AlexNet: 

- We will train the model on CIFAR-10, not ImageNet, so the images put into the model will be 32x32x3, not 224x224x3.
    
- We will not implement Local Response Normalizaiton to the convolutional layers, which was originally in the paper. It is not commonly used anymore. Instead, we will apply 2D Batchnorm to each of the layers.
    
- For data augementation, we will only:
    
    - Randomly crop the images.
        
    - Randomly flip the images horizontall;y, and instead of the probabilty being 50%, it will instead be all of the images.
        
    - We will not implement color normalization on the training set.

#### VGGNet:

Paper: https://arxiv.org/pdf/1409.1556.pdf

**" Very Deep Convolutional Networks For Large-Scale Image Recognition"**
    ---*Karen Simonytan, Andrew Zisserman*

In the traditional paper, Karen and Andrew demonstrated 5 different model architectures, being named A-E:

- A: 11 Layers, 8 of which being convolutional layers.

- A-LRN: 11 Layers, with Local Response Normalization included in the first convolution layer.

- B: 13 Layers, 10 of which being convolutional layers.

- C: 16 Layers, 13 of which being convolutional layers.

- D: 16 Layers, 13 of which being convolutional layers, the difference from C being that they modified the size of some of the layers.

- E: 19 Layers, 16 of which being convolutional layers.

- In each of the model architectures, they incorporated 1 x 1 convolutional layers, as a way to increase the non-linearity of the decision function without affecting the receptive fields of the convolutional layers.

- While it did not win the ImageNet challenge, it was 2nd place in 2014 (GoogleNet won in 2014).

- Data augmentation implemented on the images included:

    - Random scaling and then cropping of the images.

    - Random horizontal flipping of the images.

    - Color jittering of the images, which means that the RGB color channels of the images were altered by adding a random value drawn from a normal distribution with zero mean and 0.1 standard deviation.

    - Also, the paper mentioned the technique of using multi-crop evaluation at test time, meaning that they took multiple 224x224 crops from the test image and averaged the predictions made by the network on these aforementioned crops.

In this modified version of VGGNet, we will:

- Only implement the smallest of the demonstrated model architecture, that being the model A, which consists of 11 layers.

- For data augmentation, we will:

    - Randomly crop the images, we will not implement scaling of the images.

    - Randomly flip the images horizontally, flipping all of the images in the training set.

    - We will not implement multi-crop evaluations of the images at test time.

#### GoogleNet

Paper: https://arxiv.org/abs/1409.4842v1

**"Going Deeper With Convolutions"**
    ---*Christian Szegedy, Wei Liu, Yangqing Jia, et al.*

In this paper, the authors developed a new neural network architecture called the 'Inception Module', which improved the utilization of computing resources inside of said network. This novel idea increased the performance of the model on the ImageNet dataset and won the ILSVRC in 2014.

The Inception Module was inspired by the "Network in Network" paper, also published in 2014 (https://arxiv.org/pdf/1312.4400.pdf). This paper described, 'building micro neural networks with more complex structures to abstract the data within the receptive field' (page 1). This was to be an improvement to simply using linear filters followed by non-linear functions.

In the paper by Szegedy et al., the original inception module consists of four parallel branches:

- 1x1 convolution, 3x3 convolution, 5x5 convolution, and a 3x3 max pool.

- Each of the branches had a different number of filters, and these computations were followed by a batch norm and a ReLU.

- Outputs of these branches were concatenated along the channel dimension.

- That concatenated output will then be sent to the next layer.

- They improved the modules by placing 1x1 convolutions before the 3x3 and 5x5 to reduce the dimensionality of the input.

- They also replaced the 5x5 convolution with two consecutive 3x3 convolutions, which will produce the same output, but will involve fewer parameters, thus making it more efficient.

An important thing to know that has been outlined in the paper is that 'as features of higher the number of channels increases as we go deeper into the network. This allows the model to capture more high-level features as it progresses through the layers.

In this modified version of GoogleNet, we will:

- Implement a smaller version of the original GoogleNet, which is more suitable for the CIFAR-10 dataset.

- Retain the core architecture of the Inception Module but adjust the number of filters in the branches to better accommodate the smaller images.

- For data augmentation, we will:

    - Randomly crop the images.

    - Randomly flip the images horizontally.

    - We will not implement any color jittering or scaling.

### ResNet

Paper: https://arxiv.org/abs/1512.03385

**"Deep Residual Learning for Image Recognition"**
    --- *Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun*

The authors of this paper introduced the idea of residual learning, which is based on adding shortcut connections between layers in the network. These connections allow the model to learn a residual function, which represents the difference between the input and the output of a stack of layers. The residual connections are added to the input, allowing the network to learn the identity function H(x) = x easier. This allieviates the issue of vanishing / exploding gradients. 

Identity mappings are ways of designing the skip connections and the activation functions in ResNet such that the input be directly propagated to the output without any modification. So, the network can learn the residual function F(x) with respect to the input x, instead of learning a new representation H(x). This way, the network can avoid degradation problems when increasing the depth, and can also benefit from shortcut paths that allow information to flow smoothly across layers.

Lets explain in more detail:

Residual function:

- The residual function F(x, {Wi}) represents the mapping that needs to be learned. It is the difference between the input x and the output y of the layers considered. The equation y = F(x, {Wi}) + x shows that the output y is the sum of the input x and the residual function F(x, {Wi}). In the example with two layers, F = W2σ(W1x), where σ denotes the ReLU activation function.

Shortcut connection:

- The shortcut connection is used to perform the operation F + x, which is an element-wise addition. It enables the network to learn the identity function more easily, helping to overcome the vanishing gradient problem in deep networks. The authors use a second nonlinearity (ReLU) after the addition.

No extra parameters or computation complexity:

- The shortcut connections do not introduce any additional parameters or computational complexity, making the architecture efficient and allowing for fair comparisons between plain and residual networks.

Matching dimensions:

- The dimensions of x and F must be equal for the element-wise addition. If they are not, a linear projection Ws can be used in the shortcut connection to match the dimensions: y = F(x, {Wi}) + Ws x. The authors found that using the identity mapping (no matrix Ws) is sufficient for addressing the degradation problem and is more computationally efficient.

Flexibility of residual function:

- The residual function F can have different forms, such as two or three layers (as shown in Fig. 5 of the paper). However, if F has only one layer, the equation becomes similar to a linear layer, and the authors did not observe any advantages in that case.

Applicability to convolutional layers:

- The notations used in the paper can also be applied to convolutional layers. The function F(x, {Wi}) can represent multiple convolutional layers, and the element-wise addition is performed on the feature maps, channel by channel.
In the traditional ResNet, the authors proposed four variations of the architecture with 18, 34, 50, 101, and 152 layers. They achieved state-of-the-art performance on the ImageNet dataset in 2015 and won the ILSVRC competition.

In this modified version of ResNet, we will:

- Implement the 34-layer variant of the proposed architecture, to better accommodate the CIFAR-10 dataset. This is chosen because in the paper it was demonstrated that the performance was not substantial between the 18-layer variant and the 18-layer plain net (although the res net version do converge faster).

    -- please note that the higher layer version are "better than the 18 and 34 versions by considerable margins" (page 7).

- Retain the core architecture of the residual connections and building blocks.

- For data augmentation, we will:

    - Randomly crop the images.

    - Randomly flip the images horizontally.

    - We will not implement any color jittering or scaling.


### How to run the project:

To train and test the models, simply run the following command:

```bash
python main.py --model {model_name} --lr {learning_rate} --epoch {num_epochs} --trainbatchsize {train_batch_size} --testbatchsize {test_batch_size}


