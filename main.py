# Importation of modules/packages.
import torch.utils.data
import torch.optim as optim
import torchvision
import torch.backends.cudnn as cudnn
from torchvision import transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import argparse
from Models import *
from misc import misc_bar
from tqdm import tqdm
import time

# Main function to execute model. Consists of parser that includes parameters used for the training of 
# the model (learning rate, epoch, etc.)
def main():
    parser = argparse.ArgumentParser(description="cifar-10 with PyTorch")
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epoch', default=100, type=int, help='number of epochs tp train for')
    parser.add_argument('--trainBatchSize', default=100, type=int, help='training batch size')
    parser.add_argument('--testBatchSize', default=100, type=int, help='testing batch size')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='whether cuda is in use')
    parser.add_argument('--model', default='AlexNet', choices=['AlexNet','VGGNet','GoogleNet','ResNet', 'SqueezeNet', 'DenseNet'],help='choose the model to use: AlexNet, VGGNet, GoogleNet, ResNet, SqueezeNet, or DenseNet')
    parser.add_argument('--plotting', default=False, help='choose to plot or not to plot the images thats transformed.')
    args = parser.parse_args()

    # Data setup class. Used to set up the data used for the model.
    setup = DataSetup(args, args.model)
    train_loader, test_loader = setup.load_data()

    # Lets show the plotting of images.
    if args.plotting:
        plot = Plotting()
        plot.plot_normal()
        plot.plot_flipped()
        plot.plot_cropped()

        # VGGNet involves color jittering. If using this model, show the plot of color jittered images.
        if args.model == 'VGGNet':
            plot.plot_jittered()

    # Lets create an instance of the solver class, then we can execute said solver to train and test the model.
    solver = Solver(args, train_loader, test_loader)
    solver.execute(setup.load_data())

# This class is used for setting up the data:
# Load the data, apply some transformations, set up the training and testing set.
class DataSetup(object):
    def __init__(self, config, model_name):
    # Apply transformations to the images, according to various model architectures.
    # Then, set up the training set and testing sets for usage.
    # At each iteration, it will take batch size amount of random images, put into the model, repeat.
    # iteration per epoch is the total training images / batch size.
        self.model_name = model_name
        self.train_batch_size = config.trainBatchSize
        self.test_batch_size = config.testBatchSize

    # Note that we will have to subsample the training set in order to train the model faster. We will only use 10% of the training and testing sets.
    def load_data(self):
        # Loading and image transformations of the datasets.
        # First, in order to use .Normalize() properly, we will ahve to compute the mean and standard deviation of out subsampled training and testing sets.

        # Lets change the images into a tensor first.
        tensor_transform = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=tensor_transform)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=tensor_transform)

        # Lets do that subsampling of the sets. 10%. In total, 5000 training samples, 1000 testing samples.
        # Get the indices for the subsampling, then making the new training and testing set based on these indices.
        sub_train_size = int(len(train_set)*0.1)
        sub_test_size = int(len(test_set)*0.1)
        train_indices = np.random.choice(len(train_set), sub_train_size, replace=False)
        test_indices = np.random.choice(len(test_set), sub_test_size, replace=False)
        mini_train = torch.utils.data.Subset(train_set, train_indices)
        mini_test = torch.utils.data.Subset(train_set, test_indices)
        
        # We must compute the mean and std of the subsampled images along the channels of said images. Then we can normalize using these values.
        # Lets first stack all of the subsampled iamges together to make the computation of the mean and std easier.
        train_images =  torch.stack([image for image, _ in mini_train], dim=0)
        test_images = torch.stack([image for image, _ in mini_test], dim=0)

        # Now, we can calculation of the mean and standard deviation of the images in each respective sets.
        train_mean  = train_images.mean(dim=[0,2,3])
        train_std = train_images.std(dim=[0,2,3])
        test_mean = test_images.mean(dim=[0,2,3])
        test_std = test_images.std(dim=[0,2,3])

        # Now we can apply transformations to our sets (including normalization), based on the model architecture's needs.
        if self.model_name == "AlexNet":
            # For AlexNet, apply horizontal flips and random crops, then normalize.
            train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(size=32, padding=4), transforms.ToTensor(), 
                                                  transforms.Normalize(train_mean, train_std)])
            test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(test_mean, test_std)])
        
        elif self.model_name == "VGGNet" or self.model_name == "GoogleNet" or self.model_name == 'ResNet' or self.model_name == 'SqueezeNet' or self.model_name == 'DenseNet':
            # For VGGNet, GoogleNet, ResNet, or SqueezeNet, applying horizontal flips, random crops, as well as color jitterng. Then normalize.
            # For the color jittering, lets set the brightness as .50, and the hue as .30.
            train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(size=32, padding=4), transforms.ColorJitter(brightness=.5, hue=.3),
                                                   transforms.ToTensor(), transforms.Normalize(train_mean, train_std)])
            test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(test_mean, test_std)])
        
        # Apply transformations to dataset, then we can finally subsample based on same indices as before, and everything will be normalized correctly based on that subsampled sets.
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform) 
        mini_train = torch.utils.data.Subset(train_set, train_indices)
        mini_test = torch.utils.data.Subset(train_set, test_indices)

        # Train, test, dataloader.
        self.train_loader = torch.utils.data.DataLoader(dataset=mini_train, batch_size=self.train_batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=mini_test, batch_size=self.test_batch_size, shuffle=False)
        return self.train_loader, self.test_loader
        
        

            
# This class is used for plotting of the images in the dataset.
# Used to demonstrate what is in the image, and the various transformations that is applied to said images.
class Plotting():
    # Lets get the CIFAR-10 data. We will not apply transforms to it, but instead do it manually in the plotting.
    def __init__(self):
        self.train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_set, batch_size=None, shuffle=False)
        self.fixed_images, self.fixed_labels = self._get_fixed_images(self.train_loader)      

    # Un-transformed images.
    def plot_normal(self):
        self._plot_transformed_images("Normal Images", None)

    # Images that have been flipped.
    def plot_flipped(self):
        flip_transform = transforms.RandomHorizontalFlip(p=1.0)
        self._plot_transformed_images("Flipped Images", flip_transform)

    # Images that have been cropped.
    def plot_cropped(self):
        crop_transform = transforms.RandomCrop(size=32, padding=4)
        self._plot_transformed_images("Cropped Images", crop_transform)

    # Images that have been color-jittered.
    def plot_jittered(self):
        jittered_transform = transforms.ColorJitter(brightness=.5, hue=.3)
        self._plot_transformed_images("Color Jittered Images", jittered_transform)

    def _get_fixed_images(self, train_loader):
        train_images = []
        train_labels = []

        # For each image and label in train_loader, we will change it to an array.
        for img, labels in train_loader:
            train_images.append(np.asarray(img))
            train_labels.append(labels)

        X_train = np.array(train_images)
        y_train = np.array(train_labels)

        fixed_images = []
        fixed_labels = []


        for y in range(10):
            idxs = np.flatnonzero(y_train == y)
            fixed_indices = idxs[:7]
            fixed_images.extend(X_train[fixed_indices])
            fixed_labels.extend(y_train[fixed_indices])

        return np.array(fixed_images), np.array(fixed_labels)

    def _plot_transformed_images(self, plot_title, transform):
        X_train = self.fixed_images
        y_train = self.fixed_labels

        classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        num_classes = len(classes)
        samples_per_class = 7

        plt.figure(figsize=(10, 7))
        plt.suptitle(plot_title, fontsize=16)

        for y, cls in enumerate(classes):
            idxs = np.flatnonzero(y_train == y)
            for i, idx in enumerate(idxs):
                plt_idx = i * num_classes + y + 1
                plt.subplot(samples_per_class, num_classes, plt_idx)

                img = X_train[idx]

                # Apply transformation if specified
                if transform is not None:
                    img_pil = transforms.ToPILImage()(img)
                    img_transformed = transform(img_pil)
                    img_numpy = np.asarray(img_transformed)
                else:
                    img_numpy = img

                plt.imshow(img_numpy)
                plt.axis('off')
                if i == 0:
                    plt.title(cls)
        plt.show()

# Solver class. This initializes parameters, loads and preprocessed the data, then train.
class Solver(object):
    def __init__(self, config, train_loader, test_loader):
        self.model = None
        self.lr = config.lr
        self.epochs = config.epoch
        self.train_batch_size = config.trainBatchSize
        self.test_batch_size = config.testBatchSize
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.cuda = config.cuda
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model_name = config.model

    # Lets load the model.
    def load_model(self):
        if self.cuda:
            self.device = torch.device('cuda')
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        # Pick model to run ! If not using model, just comment it out:
        if self.model_name == "AlexNet":
            self.model = AlexNet(num_classes=10).to(self.device)  # AlexNet
        elif self.model_name == "VGGNet":
            self.model = VGGNet(num_classes=10).to(self.device)  # VGGNet
        elif self.model_name == "GoogleNet":
            self.model = GoogleNet(num_classes=10).to(self.device) # GoogleNet
        elif self.model_name == 'ResNet':
            self.model = ResNet(ResidualBlock, [3, 4, 5,3], num_classes=10).to(self.device) # ResNet
        elif self.model_name == 'SqueezeNet':
            self.model = SqueezeNet(num_classes=10).to(self.device) # SqueeezeNet
        elif self.model_name == 'DenseNet':
            self.model = DenseNet(num_classes=10).to(self.device) #DenseNet

        # Optimization. Adam is great for such tasks.
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        # Lets decay the learning rate after epoch number(s) is reached.
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[75, 150], gamma=0.5)
        # Finally, computation of loss (softmax and categorical cross entropy of course :) )
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    # Function that trains the model.
    def train(self):
        print("Train:")
        self.model.train() # .model is now in train mode (sets up layer criteria for some layers in the net).
        train_loss = 0
        train_correct = 0
        total = 0

        # Progress bar, to see progress of the training. 
        progressbar = tqdm(self.train_loader, desc="Training", unit="batch")
        # Lets enumerate through the inputs and labels in the set
        for batch_num, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)   
            self.optimizer.zero_grad()                  # Set the gradient to zero.                 
            output = self.model(inputs)                 # Send the input data into to the model.
            loss = self.criterion(output, targets)       # Calculation of the loss.
            loss.backward()                             # Backprop to get the gradients.
            self.optimizer.step()                       # Optimizer to update parameters.
            train_loss += loss.item()
            prediction = torch.max(output, 1)           # The second param "1" represents the dimension to be reduced
            total += targets.size(0)                     

            # train_correct is incremented by one if predicted right
            train_correct += np.sum(prediction[1].cpu().numpy() == targets.cpu().numpy())

            # Progress bar to see progress.
            progressbar.update()
            progressbar.set_description('Loss: %.4f | Acc: %.3f%% (%d/%d) | LR: %.4f' % (train_loss / (batch_num + 1), 100. *  train_correct / total, train_correct, total, self.optimizer.param_groups[0]['lr']))
        # Close the progress bar after trainng.
        progressbar.close()
        return train_loss, train_correct / total
    
    # Function that tests the model.
    def test(self):
        print("test:")
        self.model.eval()
        test_loss = 0
        test_correct = 0
        total = 0

        # Don't need to compute gradients at test time. Same process as the training.
        progressbar = tqdm(self.test_loader, desc="Testing", unit="batch")
        with torch.no_grad():
            for batch_num, (inputs, targets) in enumerate(self.test_loader):
                inputs, target = inputs.to(self.device), targets.to(self.device)
                output = self.model(inputs)
                loss = self.criterion(output, targets)
                test_loss += loss.item()
                prediction = torch.max(output, 1)
                total += target.size(0)
                test_correct += np.sum(prediction[1].cpu().numpy() == targets.cpu().numpy())

                progressbar.update()
                progressbar.set_description('Loss: %.4f | Acc: %.3f%% (%d/%d) | LR: %.4f' % (test_loss / (batch_num + 1), 100. *  test_correct / total, test_correct, total, self.optimizer.param_groups[0]['lr']))
        # Close the progress bar after trainng.
        progressbar.close()
        return test_loss, test_correct / total

    # Lets save the model after testing.
    def save(self):
        model_out_path = "model.pth"
        torch.save(self.model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    # Now, lets plot the results of our model.
    def plot_loss_accuracy(self, train_losses, train_accuracies, test_losses, test_accuracies):
        epochs = range(1, self.epochs + 1)

        plt.figure(figsize=(12, 5))

        # Plot training and testing loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, label='Train Loss')
        plt.plot(epochs, test_losses, label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss vs. Epochs')
        plt.legend()

        # Plot training and testing accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accuracies, label='Train Accuracy')
        plt.plot(epochs, test_accuracies, label='Test Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs. Epochs')
        plt.legend()

        plt.tight_layout()
        plt.show()

    # Lets run a complete training and testing loop for a specified number of epochs.
    def execute(self, setup):
        train_loader, test_loader = setup
        self.load_model()
        train_losses = []
        train_accuracies = []
        test_losses = []
        test_accuracies = []

        best_accuracy = 0
        for epoch in range(1, self.epochs + 1):
            self.scheduler.step(epoch)
            print(f"\n-----------> Epoch: {epoch}/{self.epochs}")

            train_loss, train_accuracy = self.train()
            test_loss, test_accuracy = self.test()

            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)

            best_accuracy = max(best_accuracy, test_accuracy)
            if epoch == self.epochs:
                print("===> BEST TEST ACCURACY: %.3f%%" % (best_accuracy * 100))
                self.save()

        self.plot_loss_accuracy(train_losses, train_accuracies, test_losses, test_accuracies)


if __name__ == '__main__':
    main()
