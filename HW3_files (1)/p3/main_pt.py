import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import argparse
import matplotlib.pyplot as plt
from torchsummary import summary
import numpy as np

# Argument parser
parser = argparse.ArgumentParser(description='EE379K HW3 - Starter PyTorch code')
# Define the mini-batch size, here the size is 128 images per batch
parser.add_argument('--batch_size', type=int, default=128, help='Number of samples per mini-batch')
# Define the number of epochs for training
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
# Define the model class
parser.add_argument('--model_class', type=str, default="VGG11", help='Model to train')
args = parser.parse_args()

# Always make assignments to local variables from your args at the beginning of your code for better
# control and adaptability
num_epochs = args.epochs
batch_size = args.batch_size
model_to_train = args.model_class

# Each experiment you will do will have slightly different results due to the randomness
# of the initialization value for the weights of the model. In order to have reproducible results,
# we have fixed a random seed to a specific value such that we "control" the randomness.
random_seed = 1
torch.manual_seed(random_seed)

# CIFAR10 Dataset (Images and Labels)
train_dataset = dsets.CIFAR10(root='data', train=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
]), download=True)

test_dataset = dsets.CIFAR10(root='data', train=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
]))



# Dataset Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# for i, (images, labels) in enumerate(train_loader):
#     print(images.shape)

if (model_to_train == "VGG11"):
    from models.vgg11_pt import VGG
    model_var = VGG
elif (model_to_train == "VGG16"):    
    from models.vgg16_pt import VGG
    model_var = VGG
elif (model_to_train == "Mobilenetv1"):    
    from models.mobilenet_pt import MobileNetv1
    model_var = MobileNetv1


# TODO: Get VGG11 model
model = model_var()

# TODO: Put the model on the GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # use cuda if available
model.to(device)        # put model to cuda


summary(model, (3, 32, 32))

# Define your loss and optimizer
criterion = nn.CrossEntropyLoss()  # Softmax is internally computed.
optimizer = torch.optim.Adam(model.parameters())


starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

timing_arr = np.zeros(num_epochs)

test_acc_list = []

# Training loop
for epoch in range(num_epochs):

    starter.record()    # start count time


    # Training phase
    train_correct = 0
    train_total = 0
    train_loss = 0
    # Sets the model in training mode.
    model = model.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
        # TODO: Put the images and labels on the GPU
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


    ender.record()  # end count time

    torch.cuda.synchronize()             # cuda is asynchronous, wait for synq
    
    curr_time = starter.elapsed_time(ender)
    
    timing_arr[epoch] = curr_time

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
            # TODO: Put the images and labels on the GPU
            images, labels = images.to(device), labels.to(device)       #put images and labels to gpu

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
    print('Test loss: %.4f Test accuracy: %.2f %%' % (test_loss / (batch_idx + 1),100. * test_correct / test_total))
    test_acc_list.append(100. * test_correct / test_total)


total_training_time = np.sum(timing_arr)

print("Total training time = %f sec" %(total_training_time/1000))

# TODO: Save the PyTorch model in .pt format
if (model_to_train == "VGG11"):
    torch.save(model.state_dict(), '/work/08382/etaka/maverick2/HW3_files/models/trained_VGG11.pt')
elif (model_to_train == "VGG16"):
    torch.save(model.state_dict(), '/work/08382/etaka/maverick2/HW3_files/models/trained_VGG16.pt')
elif (model_to_train == "Mobilenetv1"):
    torch.save(model.state_dict(), '/work/08382/etaka/maverick2/HW3_files/models/trained_Mobilenetv1.pt')


plt.plot(test_acc_list, label='Testing Accuracy')
plt.title('Testing Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy[%]')
plt.legend()
plt.grid()
if (model_to_train == "VGG11"):
    plt.savefig("VGG11.png")
elif (model_to_train == "VGG16"):
    plt.savefig("VGG16.png")
elif (model_to_train == "Mobilenetv1"):
    plt.savefig("Mobilenetv1.png")