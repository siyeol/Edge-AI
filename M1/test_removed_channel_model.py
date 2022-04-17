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
# from mobilenet_rm_filt_pt import MobileNetv1, remove_channel
# from mobilenet_rm_filt_pt import remove_channel

batch_size = 128
fine_tune_epochs = 2
learning_rate = 0.001
enable_cuda = True
load_my_model = True
pruning_method = "chn_prune"
fine_tune = True



if (torch.cuda.is_available()) and enable_cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')



def remove_channel(input_model):
    '''
    Input: model
           description: the pruned model
    Ouput: new_model
           description: the new model generating by removing all-zero channels
    '''
    
    new_model = copy.deepcopy(input_model)
    score_list = torch.sum(torch.abs(new_model.conv1.weight.data), dim=(1,2,3))
    next_layer_score_list = torch.sum(torch.abs(new_model.layers[0].conv1.weight.data), dim=(1,2,3))
    score_list = score_list * next_layer_score_list
    out_planes_num = int(torch.count_nonzero(score_list))
    out_planes_idx = torch.squeeze( torch.nonzero(score_list, as_tuple=False))
    conv1_wgt=copy.deepcopy(new_model.conv1.weight.data)
    new_model.conv1 = nn.Conv2d(3, out_planes_num, kernel_size=3, stride=1, padding=1, bias=False)
    new_model.bn1 = nn.BatchNorm2d(out_planes_num)
    new_model.conv1.weight.data[:,:,:,:] = conv1_wgt[out_planes_idx,:,:,:]

    in_planes_num = out_planes_num
    in_planes_idx = out_planes_idx
    for i,block in enumerate(new_model.layers):
        conv1_wgt=copy.deepcopy(block.conv1.weight.data)
        new_model.layers[i].conv1 = nn.Conv2d(in_planes_num, in_planes_num, kernel_size=3, stride=block.stride,
                                              padding=1, groups=in_planes_num, bias=False)
        new_model.layers[i].bn1 =  nn.BatchNorm2d(in_planes_num)          
        new_model.layers[i].conv1.weight.data[:,:,:,:] = conv1_wgt[in_planes_idx,:,:,:]
        score_list = torch.sum(torch.abs(block.conv2.weight.data), dim=(1,2,3))
        if i <len(new_model.layers)-1:
            next_layer_score_list = torch.sum(torch.abs(new_model.layers[i+1].conv1.weight.data), dim=(1,2,3))
            score_list = score_list * next_layer_score_list
        out_planes_num = int(torch.count_nonzero(score_list))
        out_planes_idx = torch.squeeze( torch.nonzero(score_list, as_tuple=False))
        conv2_wgt=copy.deepcopy(block.conv2.weight.data)
        new_model.layers[i].conv2 = nn.Conv2d(in_planes_num, out_planes_num, kernel_size=1, stride=1,
                                              padding=0, bias=False)
        new_model.layers[i].bn2 = nn.BatchNorm2d(out_planes_num)


        for idx_out,n in enumerate(out_planes_idx):

            new_model.layers[i].conv2.weight.data[idx_out,:,:,:] = conv2_wgt[n,in_planes_idx,:,:]
        in_planes_num = out_planes_num
        in_planes_idx = out_planes_idx
    lin_wgt=copy.deepcopy(new_model.linear.weight.data)
    lin_bias=copy.deepcopy(new_model.linear.bias.data)
    new_model.linear = nn.Linear(in_planes_num, new_model.num_classes)


    new_model.linear.weight.data = lin_wgt[:,out_planes_idx]    
    new_model.linear.bias.data = lin_bias
    return new_model
    


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


# model = MobileNetv1()


# model.load_state_dict(torch.load("5percent_pruned_zeroelim.pt"))
# torch.load("5percent_pruned_zeroelim.pt")
# model = model.to(torch.device('cuda'))
# summary(model, (3, 32, 32))

# model = torch.load("5my_model.pt")
# model = model.to(torch.device('cuda'))

# summary(model, (3, 32, 32))

# test(1, "thres", 0.05)


# model = torch.load("5my_model.pt")
# model = model.to(torch.device('cuda'))

# summary(model, (3, 32, 32))
# test(1, "thres", 0.05)





pruning_fraction = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.90]
# pruning_fraction = [0.4]

for prune_thres in pruning_fraction:

    filepath = str(int(100*prune_thres))+"percent_pruned_only.pt"
    model = torch.load(filepath)
    model = model.to(torch.device('cuda'))
    
    print("\n")
    print("Only pruned accuracy:")
    acc = test(1, "thres", prune_thres)

    criterion = nn.CrossEntropyLoss()
    criterion_sum = nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    iteration = 0

    model = remove_channel(model)
    model = model.to(torch.device('cuda'))
    summary(model, (3, 32, 32))

    print("\n")
    print("New accuracy after remove channel function:")
    for fine_tune_epoch in range(fine_tune_epochs):
        train(fine_tune_epoch)
        acc = test(1, "thres", prune_thres)

    filepath_new = str(int(100*prune_thres))+"percent_pruned_zeroelim.pt"   
    torch.save(model, filepath_new)
 


# model = remove_channel(model)
# model = model.to(torch.device('cuda'))


# model = remove_channel(model)
# model = model.to(torch.device('cuda'))
# # summary(model, (3, 32, 32))
# test(1, "thres", 0.05)

# for fine_tune_epoch in range(fine_tune_epochs):
#         train(fine_tune_epoch)
#         test(1, "thres", 0.05)
    


# model = remove_channel(model)
# model = model.to(torch.device('cuda'))
# # summary(model, (3, 32, 32))

# test(1, "thres", 0.05)