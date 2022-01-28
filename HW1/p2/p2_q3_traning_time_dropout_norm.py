import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt
import time
 
 
# Argument parser
parser = argparse.ArgumentParser(description='EE379K HW1 - SimpleFC')
# Define the mini-batch size, here the size is 128 images per batch
parser.add_argument('--batch_size', type=int, default=128, help='Number of samples per mini-batch')
# Define the number of epochs for training
parser.add_argument('--epochs', type=int, default=25, help='Number of epoch to train')
# Define the learning rate of your optimizer
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
args = parser.parse_args()
 
# The size of input features
input_size = 28 * 28
# The number of target classes, you have 10 digits to classify
num_classes = 10
 
# Always make assignments to local variables from your args at the beginning of your code for better
# control and adaptability
num_epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.lr
 
# Each experiment you will do will have slightly different results due to the randomness
# of the initialization value for the weights of the model. In order to have reproducible results,
# we have fixed a random seed to a specific value such that we "control" the randomness.
random_seed = 1
torch.manual_seed(random_seed)
 
# MNIST Dataset (Images and Labels)
train_dataset = dsets.MNIST(root='data', train=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=0.1307, std=0.3081),]), download=True)
test_dataset = dsets.MNIST(root='data', train=False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=0.1307, std=0.3081),]))
 
# Dataset Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
 
dropout_prob = 0.5
# Define your model
class SimpleFC(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleFC, self).__init__()
        self.linear1 = nn.Linear(input_size, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, num_classes)
        self.drop_layer = nn.Dropout(p=dropout_prob)
       
 
    # Your model only contains a single linear layer
    def forward(self, x):
        out = F.relu(self.linear1(x))
        out = self.drop_layer(out)
        out = F.relu(self.linear2(out))
        out = self.drop_layer(out)
        out = F.relu(self.linear3(out))
        out = self.drop_layer(out)
        out = self.linear4(out)
        return out
 
 
model = SimpleFC(input_size, num_classes)
 
# Define your loss and optimizer
criterion = nn.CrossEntropyLoss()  # Softmax is internally computed.
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
 
 
 
loss_train_his = []
loss_test_his = []
 
acc_train_his = []
acc_test_his = []
 
 
total_training_time = 0


# Training loop
for epoch in range(num_epochs):

    start_training_epoch = time.time()   # start count time for each epoch


    # Training phase
    train_correct = 0
    train_total = 0
    train_loss = 0
    # Sets the model in training mode.
    model = model.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Here we vectorize the 28*28 images as several 784-dimensional inputs
        images = images.view(-1, input_size)
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
   

    end_training_epoch = time.time()     # end timing measurement

    elapsed_training_time = end_training_epoch - start_training_epoch

    total_training_time += elapsed_training_time
       
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
            # Here we vectorize the 28*28 images as several 784-dimensional inputs
            images = images.view(-1, input_size)
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
   
 

print("Total training time = %f sec" %total_training_time)


loss_plot = plt.figure(1)
 
plt.plot(loss_train_his, 'g', label='Training loss')
plt.plot(loss_test_his, 'b', label='Test loss')
plt.title('Training and Test loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
 
plt.savefig("p2_q2_loss_output_00.jpg")
 
 
# acc_plot = plt.figure(2)
 
# plt.plot(acc_train_his, 'g', label='Training Accuracy')
# plt.plot(acc_test_his, 'b', label='Test Accuracy')
# plt.title('Training and Test Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
 
# plt.savefig("acc_outp
