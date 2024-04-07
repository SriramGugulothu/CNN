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
import torchvision.models as models
import argparse
# imported the libraries needed.
#!pip install wandb  #commented since I have installed the wandb 
import wandb
from wandb.keras import WandbCallback
import socket
socket.setdefaulttimeout(30)
wandb.login()
wandb.init(project ='DL_assignment_2_B') # connected to partB project in wandb

# data processing

transform = transforms.Compose([
    transforms.Resize((224,224)), #reshaped the data to be used by RESNET50
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))]) #normalized the data

train_dataset = datasets.ImageFolder(root='inaturalist_12K/train',transform=transform)
train_dataset,val_dataset = torch.utils.data.random_split(train_dataset,[8000,1999]) #splitting data into 80%training and 20% validation data

transform2 = transforms.Compose([ # transform for augumentation of data
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    transforms.RandomRotation(10),      # Randomly rotate the image by a maximum of 10 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Adjust brightness, contrast, saturation, and hue
    transforms.RandomResizedCrop(224),  # Randomly crop and resize the image to 224x224
    transforms.ToTensor(),              # Convert the image to a PyTorch tensor
    transforms.Normalize((0.5,),(0.5,))  # Normalize the image
])

train_dataset2 = datasets.ImageFolder(root='inaturalist_12K/train',transform=transform2)
train_dataset_aug,val_dataset_aug = torch.utils.data.random_split(train_dataset2,[8000,1999]) #80%training data, 20% validation data

def dataFun(aug,batchSize): #function to return the data loaders depending on bacth size and augmentation
    if(aug == 'no'):
        train_loader = torch.utils.data.DataLoader(train_dataset,batch_size =batchSize,shuffle = True,num_workers=2,pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset,batch_size =batchSize,shuffle = True,num_workers=2,pin_memory=True)
        return train_loader,val_loader
    else:
        train_loader_aug = torch.utils.data.DataLoader(train_dataset_aug,batch_size =batchSize,shuffle = True,num_workers=4,pin_memory=True)
        val_loader_aug = torch.utils.data.DataLoader(val_dataset_aug,batch_size =batchSize,shuffle = True,num_workers=4,pin_memory=True)
        return train_loader_aug,val_loader_aug
    
def RESNET50(NUM_OF_CLASSES): # this function returns the model by freezing all but not last layer
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, NUM_OF_CLASSES) # modifying output layer to 10 neurons 
    
    for param in model.parameters(): # freezing
        param.requires_grad = False
        
    for param in model.fc.parameters(): #unfreezing
        param.requires_grad = True
   
    return model

def RESNET50_1(k,NUM_OF_CLASSES): #this function returns the model by freezing first k layers only
    model = models.resnet50(pretrained=True)    
    
    params = list(model.parameters())
    for param in params[:k]:
        param.requires_grad = False #freezing
        
    num_ftrs = model.fc.in_features
    
    model.fc = torch.nn.Linear(num_ftrs, NUM_OF_CLASSES)
    
    return model

def RESNET50_2(neurons_dense,NUM_OF_CLASSES): #this function returns the model by freezing all but not last layer after adding dense layer
    
    model = models.resnet50(pretrained=True)    
    
    activation_function_layer = nn.ReLU()
    
    for params in model.parameters():
        params.requires_grad = False #freezing
        
    num_ftrs = model.fc.in_features
    
    model.fc = nn.Sequential(
      nn.Linear(num_ftrs,neurons_dense), #adding dense layer
      activation_function_layer,
      nn.Dropout(0.4),
      nn.Linear(neurons_dense, 10)
    )

    for param in model.fc.parameters():
        param.requires_grad = True  #unfreezing
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # integrating with gpu

#training function 
def train_fun(batchSize,num_epochs,learning_rate,aug,strategy,NUM_OF_CLASSES):

    train_loader,val_loader = dataFun(aug,batchSize)  # loading training data and validation data
    if(strategy == 0): # using models
        model = RESNET50(NUM_OF_CLASSES).to(device)
    elif(strategy == 1):
        model = RESNET50_1(10,NUM_OF_CLASSES).to(device)
    else:
        model = RESNET50_2(256,NUM_OF_CLASSES).to(device)
        
        
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)#optimizer
    
    criterion = nn.CrossEntropyLoss() #cross entropy function
    
    for epoch in range(num_epochs):
        for batchId, (data, targets) in enumerate(tqdm(train_loader)):
            # Get data to cuda if possible
            data = data.to(device=device)
            targets = targets.to(device=device)
            # forward
            scores = model(data)
            loss = criterion(scores,targets)
            # backward
            optimizer.zero_grad() # making gardient = 0
            loss.backward() # calculates the gradients
            optimizer.step() #updates the parameters

        train_accuracy,train_loss = check_accuracy(train_loader, model,criterion,batchSize)
        validation_accuracy,validation_loss = check_accuracy(val_loader, model,criterion,batchSize)
        print(f"train_accuracy:{train_accuracy:.4f},train_loss:{train_loss:.4f}")
        print(f"validation_accuracy:{validation_accuracy:.4f},validation_loss:{validation_loss:.4f}")
        wandb.log({'train_accuracy':train_accuracy}) # wandb reporting
        wandb.log({'train_loss':train_loss})
        wandb.log({'val_accuracy':validation_accuracy})
        wandb.log({'val_loss':validation_loss})

def check_accuracy(loader,model,criterion,batchSize): #function to get the accuracy and cross entorpy loss 
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
            total_loss += loss.item()*batchSize #cross entropy loss 
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum().item() # correctly classified preditions
            num_samples += predictions.size(0)
    model.train()
    return (num_correct / num_samples)*100 , total_loss 

#parser code
def parse_arguments():
    parser = argparse.ArgumentParser(description='Training Parameters')

    parser.add_argument('-wp', '--wandb_project', type=str, default='DL_assignment_2_B',
                        help='Project name used to track experiments in Weights & Biases dashboard')
    
    parser.add_argument('-bS', '--batchSize', type= int, default=32, choices = [32,64],help='Choice of batch size')
    
    parser.add_argument('-e', '--epochs', type= int, default=20, choices = [10,20,30],help='Number of epochs')

    parser.add_argument('-lE', '--learnRate', type= float, default=1e-4, choices = [1e-3,1e-4],help='Learning rates')

    parser.add_argument('-ag', '--aug', type= str, default='no', choices = ['yes','no'],help='Augumentation choices')
    
    parser.add_argument('-s', '--strategy', type= int, default=1, choices = [0,1,2],help='Choice of strategies')

    return parser.parse_args()

args = parse_arguments()
wandb.init(project=args.wandb_project)

wandb.run.name=f'strategy {args.strategy}'

train_fun(args.batchSize,args.epochs,args.learnRate,args.aug,args.strategy,10)


    

