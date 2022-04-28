import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import copy
from torchsummary import summary
import argparse
# from mobilenet_rm_filt_pt import MobileNetv1, remove_channel



parser = argparse.ArgumentParser(description='M3')

parser.add_argument('--model', type=str, default="optimum0.955_pruned_KD", help='Model to test')
args = parser.parse_args()



batch_size = 128
fine_tune_epochs = 1
learning_rate = 0.001
enable_cuda = True
load_my_model = True
pruning_method = "chn_prune"
fine_tune = True



if (torch.cuda.is_available()) and enable_cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')




writer = SummaryWriter('runs/mobilenet_cifar10')


test_dataset = dsets.CIFAR10(root='data', train=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
]))

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)




def test(epoch, mode, value):
    correct = 0
    total = 0
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = Variable(images.view(-1, 3, 32, 32)).to(torch.device('cuda'))
            labels = Variable(labels).to(torch.device('cuda'))


            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

    accuracy = correct.float() / total
    writer.add_scalar("test_accuracy", accuracy, epoch)
    print(f"{mode}={value}, Test accuracy={100*accuracy.data.item():.4f}")
    return 100*accuracy.data.item()




model = torch.load(args.model + ".pt")
model = model.to(torch.device('cuda'))

summary(model, (3, 32, 32))
test(1, "model", args.model)


