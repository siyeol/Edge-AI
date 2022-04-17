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
from mobilenet_rm_filt_pt import MobileNetv1, remove_channel

batch_size = 128
fine_tune_epochs = 5
learning_rate = 0.001
enable_cuda = True
load_my_model = True
pruning_method = "chn_prune"
fine_tune = True





writer = SummaryWriter('runs/mobilenet_cifar10')
train_dataset = dsets.CIFAR10(root='data', train=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
]), download=True)

test_dataset = dsets.CIFAR10(root='data', train=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
]))

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

def load_model(model, path="trained_Mobilenetv1.pt", print_msg=True):
    try:
        model.load_state_dict(torch.load(path))
        if print_msg:
            print(f"[I] Model loaded from {path}")
    except:
        if print_msg:
            print(f"[E] Model failed to be loaded from {path}")

def model_size(model, count_zeros=True):
    total_params = 0
    nonzero_params = 0
    for tensor in model.parameters():
        t = np.prod(tensor.shape)
        nz = np.sum(tensor.detach().cpu().numpy() != 0.0)

        total_params += t
        nonzero_params += nz
    if not count_zeros:
        return int(nonzero_params)
    else:
        return int(total_params)


if (torch.cuda.is_available()) and enable_cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = MobileNetv1()
# model= nn.DataParallel(model)       # run on more gpus
model.to(device)
# model = model.to(torch.device('cuda'))

criterion = nn.CrossEntropyLoss()
# criterion_sum = nn.CrossEntropyLoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
iteration = 0

def train(epoch):
    global iteration
    model.train()
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.view(-1,3,32,32)).to(torch.device('cuda'))
        labels = Variable(labels).to(torch.device('cuda'))

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        if (i+1) % 100 == 0:
            print('Epoch: [% d/% d], Step: [% d/% d], Loss: %.4f' % (epoch+1, fine_tune_epochs, i+1, len(train_dataset)//batch_size, loss.data.item()))
        iteration += 1
    accuracy = correct.float() / total
    writer.add_scalar("train_accuracy", accuracy, epoch)
    print('Accuracy of the model on the % d train images: % f %%' % (len(train_dataset), 100*accuracy))
    return loss

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
    print(f"mode: {mode}={value}, Test accuracy={100*accuracy.data.item():.4f}")
    return 100*accuracy.data.item()


load_model(model)
# summary(model, (3, 32, 32))
test(0, mode='Non-pruned model', value='True')


def channel_fraction_pruning(model, fraction):
    mask_dict={}
    for name, param in model.state_dict().items():
        # print(name)
        if (("weight" in name) and ("conv" in name)):
            score_list = torch.sum(torch.abs(param),
                                   dim=(1,2,3)).to('cpu')
            removed_idx = []
            threshold = np.percentile(np.abs(score_list), fraction*100)
            for i,score in enumerate(score_list):
                if score < threshold:
                    removed_idx.append(i)
                param[removed_idx,:,:,:] = 0
                mask_dict[name]=torch.where(torch.abs(param) > 0,1,0)
    model.mask_dict = mask_dict

res1 = []
num_params = []





pruning_fraction = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.90]
# pruning_fraction = [0.05]

for prune_thres in pruning_fraction:
    channel_fraction_pruning(model, prune_thres)

    for fine_tune_epoch in range(fine_tune_epochs):
        train(fine_tune_epoch)
    
    model._apply_mask()
    acc = test(1, "thres", prune_thres)
    res1.append([prune_thres, acc])
    num_params.append(model_size(model, prune_thres==0))

    # model_removed_zeros = remove_channel(model)
    
    filepath = str(int(100*prune_thres))+"percent_pruned_only.pt"

    # torch.save(model.state_dict(), filepath)
    torch.save(model, filepath)




res1 = np.array(res1)

plt.figure()
plt.plot(num_params, res1[:,1])
plt.title('{}: Accuracy vs #Params'.format(pruning_method))
plt.xlabel('Number of parameters')
plt.ylabel('Test accuracy')
plt.savefig('{}_param_fine_tune_{}.png'.format(pruning_method, fine_tune))
plt.close()

plt.figure()
plt.plot(pruning_fraction, res1[:,1])
plt.title('{}: Accuracy vs Threshold'.format(pruning_method))
plt.xlabel('Pruning Threshold')
plt.ylabel('Test accuracy')
plt.savefig('{}_thresh_fine_tune_{}.png'.format(pruning_method, fine_tune))
plt.close()
