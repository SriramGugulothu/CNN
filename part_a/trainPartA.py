import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets 
import math
import argparse
# Required libraries are imported 

import wandb
from wandb.keras import WandbCallback
import socket
socket.setdefaulttimeout(30)
wandb.login()
wandb.init(project ='DL_assignment_2')
# Wadb integration

class CNN(nn.Module): # class of CNN module
    def __init__(self, in_channels=3, num_classes=10,numFilterss=32,sizeFilter=3,neurons=128,activFun='sigmoid',dropOut=0.0,batchNorm='no',org=0):
        super(CNN, self).__init__()
        self.activFunName = activFun
        self.batchNorm = batchNorm
        if(org ==  0): # the filters are same
            numFilters = [numFilterss,numFilterss,numFilterss,numFilterss,numFilterss]
        else:
            numFilters = [numFilterss,numFilterss*2,numFilterss*4,numFilterss*8,numFilterss*16] # the filters increase by *2 
        width = 0.0
        hight = 0.0
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=numFilters[0],kernel_size=sizeFilter, stride=1)
        width = (256 - sizeFilter)+1 #256 --> width of image
        hight = (256 - sizeFilter)+1 #256 --> height of image
        self.bn1 = nn.BatchNorm2d(numFilters[0])  # batch normalizations
        self.pool1 = nn.MaxPool2d(kernel_size=sizeFilter, stride=2)
        width = math.floor((width - sizeFilter)/2) + 1 # width  calculatios of feature map
        hight = math.floor((hight -sizeFilter)/2) + 1 # hight calculations of feature map
        
        self.conv2 = nn.Conv2d( in_channels=numFilters[0], out_channels=numFilters[1], kernel_size=sizeFilter,stride=1)
        width = ((width - sizeFilter))+1
        hight = ((hight-sizeFilter))+1
        self.bn2 = nn.BatchNorm2d(numFilters[1])
        self.pool2 = nn.MaxPool2d(kernel_size=sizeFilter,stride=2)
        width = math.floor((width - sizeFilter)/2) + 1
        hight = math.floor((hight -sizeFilter)/2) + 1
        
        self.conv3 = nn.Conv2d( in_channels=numFilters[1], out_channels=numFilters[2], kernel_size=sizeFilter,stride=1)
        width = ((width - sizeFilter))+1
        hight = ((hight-sizeFilter))+1
        self.bn3 = nn.BatchNorm2d(numFilters[2])
        self.pool3 = nn.MaxPool2d(kernel_size=sizeFilter,stride=2)
        width = math.floor((width - sizeFilter)/2) + 1
        hight = math.floor((hight -sizeFilter)/2) + 1
        
        self.conv4 = nn.Conv2d( in_channels=numFilters[2], out_channels=numFilters[3], kernel_size=sizeFilter,stride=1)
        width = ((width - sizeFilter))+1
        hight = ((hight-sizeFilter))+1 
        self.bn4 = nn.BatchNorm2d(numFilters[3])
        self.pool4 = nn.MaxPool2d(kernel_size=sizeFilter,stride=2)
        width = math.floor((width - sizeFilter)/2)+ 1
        hight = math.floor((hight -sizeFilter)/2) + 1
        
        self.conv5 = nn.Conv2d( in_channels=numFilters[3], out_channels=numFilters[4], kernel_size=sizeFilter,stride=1)
        width = ((width - sizeFilter))+1
        hight = ((hight-sizeFilter))+1
        self.bn5 = nn.BatchNorm2d(numFilters[4])
        self.pool5 = nn.MaxPool2d(kernel_size=sizeFilter,stride=2)
        width = math.floor((width - sizeFilter)/2) + 1
        hight = math.floor((hight -sizeFilter)/2) + 1
        
        self.dropout = nn.Dropout(p=dropOut) # added dropout to overcome overfitting.
        self.fc1 = nn.Linear(numFilters[4] * width*hight, neurons) # dense layer 
        self.bn6 = nn.BatchNorm1d(neurons)
        self.fc2 = nn.Linear(neurons,10)
        
    def forward(self, x):
        if(self.activFunName == 'relu'): # activation functions
            activation_fn = F.relu
        elif(self.activFunName == 'gelu'): 
            activation_fn = F.gelu
        elif(self.activFunName == 'silu'):
            activation_fn = F.silu
        else:
            activation_fn = F.mish

        if(self.batchNorm == 'yes'): # choosing option for batchNormalization
            x = activation_fn(self.bn1(self.conv1(x)))
        else:
            x = activation_fn(self.conv1(x))    
        x = self.pool1(x)
        
        if(self.batchNorm == 'yes'):
            x = activation_fn(self.bn2(self.conv2(x)))
        else:
            x = activation_fn(self.conv2(x))
        x = self.pool2(x)
        
        if(self.batchNorm == 'yes'):
            x = activation_fn(self.bn3(self.conv3(x)))
        else:
            x = activation_fn(self.conv3(x))
        x = self.pool3(x)
        
        if(self.batchNorm == 'yes'):
            x = activation_fn(self.bn4(self.conv4(x)))
        else:
            x = activation_fn(self.conv4(x))
        x = self.pool4(x)
        
        if(self.batchNorm == 'yes'):
            x = activation_fn(self.bn5(self.conv5(x)))
        else:
            x = activation_fn(self.conv5(x))
        x = self.pool5(x)
        
        x = x.reshape(x.shape[0], -1) # flattening the output after convolution for dense layer
        if(self.batchNorm == 'yes'):
            x = activation_fn(self.bn6(self.fc1(x)))
        else:
            x = activation_fn(self.fc1(x))
        x = self.dropout(x)  # adding drop out after dense layer for overcoming overfitting. 
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # thid code is to integrate with gpu if it is availble otherwise cpu wiil be used.

# data loading
transform = transforms.Compose([
    transforms.Resize((256,256)), # resized to a threshold value so that all images have same shape and size
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))]) # normalized for better accuracy.

train_dataset = datasets.ImageFolder(root="inaturalist_12K\train",transform=transform) # train_data loading
train_dataset,val_dataset = torch.utils.data.random_split(train_dataset,[8000,1999]) #splitting the data into 80%(training) and 20%(validation) The overall data size is 9999

transform2 = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    transforms.RandomRotation(10),      # Randomly rotate the image by a maximum of 10 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Adjust brightness, contrast, saturation, and hue
    transforms.RandomResizedCrop(256),  # Randomly crop and resize the image to 256x256
    transforms.ToTensor(),              # Convert the image to a PyTorch tensor
    transforms.Normalize((0.5,),(0.5,))  # Normalize the image
]) # for augumenting the training data
train_dataset2 = datasets.ImageFolder(root="inaturalist_12K\ train",transform=transform2)
train_dataset_aug,val_dataset_aug = torch.utils.data.random_split(train_dataset2,[8000,1999]) #  #splitting the data into 80%(training) and 20%(validation) The overall data size is 9999

test_data = datasets.ImageFolder(root="inaturalist_12K\ val",transform=transform); # test data loading.

def dataFun(aug,batchSize): # function to return the data loaders depending on augumentation.
    if(aug == 'no'):
        train_loader = torch.utils.data.DataLoader(train_dataset,batch_size =batchSize,shuffle = True,num_workers=2,pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset,batch_size =batchSize,shuffle = True,num_workers=2,pin_memory=True)
        return train_loader,val_loader
    else:
        train_loader_aug = torch.utils.data.DataLoader(train_dataset_aug,batch_size =batchSize,shuffle = True,num_workers=4,pin_memory=True)
        val_loader_aug = torch.utils.data.DataLoader(val_dataset_aug,batch_size =batchSize,shuffle = True,num_workers=4,pin_memory=True)
        return train_loader_aug,val_loader_aug
    
def train_fun(neurons,numFilters,sizeFilter,activFun,optimizerName,batchSize,dropOut,num_epochs,learning_rate,batchNorm,aug,org):

    train_loader,val_loader = dataFun(aug,batchSize)  # getting dataloaders.
    
    #test_loader = torch.utils.data.DataLoader(test_data,batch_size =batchSize,shuffle = True,num_workers=2,pin_memory=True)
    
    in_channels = 3
    num_classes = 10
    
    model = CNN(in_channels, num_classes,numFilters,sizeFilter,neurons,activFun,dropOut,batchNorm,org).to(device)
    
    if(optimizerName == 'sgd'):
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif(optimizerName == 'adam'):
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.NAdam(model.parameters(), lr=learning_rate) # optimzers selection 
        
    criterion = nn.CrossEntropyLoss() # since it is classification problem corss entropy loss is used.
    
    for epoch in range(num_epochs): # performs the training.
        for batchId, (data, targets) in enumerate(tqdm(train_loader)):
            # Get data to cuda if possible
            data = data.to(device=device)
            targets = targets.to(device=device)
            # forward
            scores = model(data) # give the last layer pre-activation values.
            loss = criterion(scores,targets) # gets the overll cross entropy loss for each batch 
            optimizer.zero_grad() # gradients are made to zero for each batch.
            loss.backward()  # calculaing the gradients
            optimizer.step() #updates the parameters
        train_accuracy,train_loss = check_accuracy(train_loader, model,criterion,batchSize) # calculates the accuracy and loss at one go
        validation_accuracy,validation_loss = check_accuracy(val_loader, model,criterion,batchSize)
        #  the below line can be uncommenteed for test accuracy and loss
        #test_accuracy,test_loss = check_accuracy(test_loader, model,criterion,batchSize)
        print(f"train_accuracy:{train_accuracy:.4f},train_loss:{train_loss:.4f}")
        print(f"validation_accuracy:{validation_accuracy:.4f},validation_loss:{validation_loss:.4f}")
        #print(f"test_accuracy:{test_accuracy:.4f},test_loss:{test_loss:.4f}")
        wandb.log({'train_accuracy':train_accuracy}) # plotting  the data in wandb
        wandb.log({'train_loss':train_loss})         
        wandb.log({'val_accuracy':validation_accuracy})
        wandb.log({'val_loss':validation_loss})

def check_accuracy(loader,model,criterion,batchSize): # function to clculate the accuracy and loss
    num_correct = 0
    num_loss = 0
    total = 0
    num_samples = 0
    total_loss = 0.0 
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            scores = model(x)
            loss = criterion(scores, y)
            total_loss += loss.item()*batchSize # sum of cross entropies
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum().item() # correctly classified data
            num_samples += predictions.size(0)
    model.train()
    return (num_correct / num_samples)*100 , total_loss

def parse_arguments():
    parser = argparse.ArgumentParser(description='Training Parameters')

    parser.add_argument('-wp', '--wandb_project', type=str, default='DL_assignment_2',
                        help='Project name used to track experiments in Weights & Biases dashboard')
    
    parser.add_argument('-n', '--neurons', type= int, default=256, choices = [128,256],help='Choice of neurons in dense layer')
    
    parser.add_argument('-nF', '--numFilters', type= int, default=64, choices = [32,64,128],help='Choice of number of filters')

    parser.add_argument('-sF', '--sizeFilter', type= int, default=5, choices = [3,5],help='Choice of kernel size')

    parser.add_argument('-aF', '--activFun', type= str, default='relu', choices = ['relu','gelu','silu','mish'],help='Choice of activation function')
    
    parser.add_argument('-opt', '--optimizer', type= str, default='nadam', choices = ['adam','nadam'],help='Choice of optimizer')

    parser.add_argument('-bS', '--batchSize', type= int, default=32, choices = [32,64,128],help='Choice of batch size')

    parser.add_argument('-d', '--dropOut', type= float, default=0.4, choices = [0,0.2,0.4],help='Choice of drop out probability')

    parser.add_argument('-nE', '--epochs', type= int, default=10, choices = [5,10],help='Choice of epochs')

    parser.add_argument('-lR', '--learnRate', type =float, default=1e-4, choices = [1e-3,1e-4],help='Choice of learnRate')

    parser.add_argument('-bN', '--batchNorm', type= str, default='yes', choices = ['yes','no'],help='Choice of batch normalization')

    parser.add_argument('-ag', '--aug', type= str, default='no', choices = ['yes','no'],help='Choice of augumentation')

    parser.add_argument('-o', '--org', type= int, default=1, choices = [0,1],help='Choice of filter organization')

    return parser.parse_args()

args = parse_arguments()
wandb.init(project=args.wandb_project)

wandb.run.name=f'activation {args.activFun}opt{args.optimizer}batchNorm{args.batchNorm}'

train_fun(args.neurons,args.numFilters,args.sizeFilter, args.activFun,args.optimizer,args.batchSize,args.dropOut,args.epochs,args.learnRate,args.batchNorm,args.aug,args.org)

    
    
    
    
    
    
    
    
  
        

