import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.functional as F
import argparse
from thop import profile
from torchsummary import summary
import matplotlib.pyplot as plt





# Argument parser
parser = argparse.ArgumentParser(description='EE379K HW1 - SimpleCNN')
# Define the mini-batch size, here the size is 128 images per batch
parser.add_argument('--batch_size', type=int, default=128, help='Number of samples per mini-batch')
# Define the number of epochs for training
parser.add_argument('--epochs', type=int, default=25, help='Number of epoch to train')
# Define the learning rate of your optimizer
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
# Define the conv1 channel size of your optimizer
parser.add_argument('--conv1_ch_size', type=int, default=32, help='Conv1 channel size')
# Define the conv2 channel size of your optimizer
parser.add_argument('--conv2_ch_size', type=int, default=64, help='Conv2 channel size')
args = parser.parse_args()

# The number of target classes, you have 10 digits to classify
num_classes = 10

# Always make assignments to local variables from your args at the beginning of your code for better
# control and adaptability
num_epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.lr
conv1_ch_size = args.conv1_ch_size
conv2_ch_size = args.conv2_ch_size

# Each experiment you will do will have slightly different results due to the randomness
# of the initialization value for the weights of the model. In order to have reproducible results,
# we have fixed a random seed to a specific value such that we "control" the randomness.
random_seed = 1
torch.manual_seed(random_seed)

# MNIST Dataset (Images and Labels)
# TODO: Insert here the normalized MNIST dataset
train_dataset = dsets.MNIST(root='data', train=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=0.1307, std=0.3081),]), download=True)
test_dataset = dsets.MNIST(root='data', train=False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=0.1307, std=0.3081),]))

# Dataset Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# Define your model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, conv1_ch_size, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(conv1_ch_size, conv2_ch_size, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear1 = nn.Linear(7 * 7 * conv2_ch_size, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.pool1(out)
        out = F.relu(self.conv2(out))
        out = self.pool2(out)
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        return out


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # use cuda if available

model = SimpleCNN(num_classes)

model.to(device)        # put model to cuda


input_rand = torch.randn(1, 1, 28, 28).cuda()

macs, params = profile(model, inputs=(input_rand, ))

print("The total number of MACS is %d" % macs)
print("The total number of PARAMS is %d" % params)


#macs, params = clever_format([macs, params], "%.3f")


summary(model, (1, 28, 28))





# Define your loss and optimizer
criterion = nn.CrossEntropyLoss()  # Softmax is internally computed.
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)


loss_train_his = []
loss_test_his = []
 
acc_train_his = []
acc_test_his = []

# Training loop
for epoch in range(num_epochs):
    # Training phase
    train_correct = 0
    train_total = 0
    train_loss = 0
    # Sets the model in training mode.
    model = model.train()
    for batch_idx, (images, labels) in enumerate(train_loader):

        images, labels = images.to(device), labels.to(device)       #put images and labels to gpu


        # Sets the gradients to zero
        optimizer.zero_grad()
        # The actual inference
        outputs = model(images)
        # Compute the loss between the predictions (outputs) and the ground-truth labels
        loss = criterion(outputs, labels)
        # Do backpropagation to update the parameters of your model
        loss.backward()
        # Performs a single optimization step (parameter update)
        optimizer.step()
        train_loss += loss.item()
        # The outputs are one-hot labels, we need to find the actual predicted
        # labels which have the highest output confidence
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
        # Print every 100 steps the following information
        if (batch_idx + 1) % 100 == 0:
            print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f Acc: %.2f%%' % (epoch + 1, num_epochs, batch_idx + 1,
                                                                             len(train_dataset) // batch_size,
                                                                             train_loss / (batch_idx + 1),
                                                                             100. * train_correct / train_total))


    loss_train_his.append(train_loss / (batch_idx + 1))
    acc_train_his.append(100. * train_correct / train_total)

    # Testing phase
    test_correct = 0
    test_total = 0
    test_loss = 0
    # Sets the model in evaluation mode
    model = model.eval()
    # Disabling gradient calculation is useful for inference.
    # It will reduce memory consumption for computations.
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):

            images, labels = images.to(device), labels.to(device)       # put also on gpu


            # Perform the actual inference
            outputs = model(images)
            # Compute the loss
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            # The outputs are one-hot labels, we need to find the actual predicted
            # labels which have the highest output confidence
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    print('Test accuracy: %.2f %% Test loss: %.4f' % (100. * test_correct / test_total, test_loss / (batch_idx + 1)))

    loss_test_his.append(test_loss / (batch_idx + 1))
    acc_test_his.append(100. * test_correct / test_total)


#torch.save(model.state_dict(), 'path/simplecnn.pth')



loss_plot = plt.figure(1)
 
plt.plot(loss_train_his, 'g', label='Training loss')
plt.plot(loss_test_his, 'b', label='Test loss')
plt.title('Training and Test loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
 
plt.savefig("p3_q2_loss_output" + "_epochs_" + str(num_epochs) + "_conv1_ch_" + str(conv1_ch_size) + "_conv2_ch_" + str(conv2_ch_size) + ".jpg")
 
 
acc_plot = plt.figure(2)
 
plt.plot(acc_train_his, 'g', label='Training Accuracy')
plt.plot(acc_test_his, 'b', label='Test Accuracy')
plt.title('Training and Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
 
plt.savefig("p3_q2_acc_output" + "_epochs_" + str(num_epochs) + "_conv1_ch_" + str(conv1_ch_size) + "_conv2_ch_" + str(conv2_ch_size) + ".jpg")