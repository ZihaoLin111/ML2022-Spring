# Import necessary packages.
import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset
from tqdm.auto import tqdm
import random
from sklearn.model_selection import KFold

_exp_name = "sample"

myseed = 4242  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)


# Transforms
test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.55807906, 0.45261728, 0.34557677], std=[0.23075283, 0.24137004, 0.24039967])
])

train_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(128, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.55807906, 0.45261728, 0.34557677], std=[0.23075283, 0.24137004, 0.24039967])
])


# Dataset class
class FoodDataset(Dataset):

    def __init__(self,path,tfm=test_tfm,files = None, mixup = False, classes = 11):
        super(FoodDataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".jpg")])
        if files is not None:
            self.files = files
        print(f"One {path} sample",self.files[0])
        self.transform = tfm

        self.mixup = mixup
        self.classes = classes

  
    def __len__(self):
        return len(self.files)
  
    def __getitem__(self,idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)
        #im = self.data[idx]
        try:
            label = int(fname.split("/")[-1].split("_")[0])
        except:
            label = -1 # test has no label

        if label == -1 or not self.mixup:
            return im,label
        
        label_a_onehot = torch.zeros(self.classes)
        label_a_onehot[label] = 1.0

        if random.random() < 0.5:
            idx2 = random.randint(0, len(self.files)-1)
            while idx2 == idx:
                idx2 = random.randint(0, len(self.files)-1)
            fname2 = self.files[idx2]
            im2 = Image.open(fname2)
            im2 = self.transform(im2)
            label2 = int(fname2.split("/")[-1].split("_")[0])
            label_b_onehot = torch.zeros(self.classes)
            label_b_onehot[label2] = 1.0

            lam = np.random.beta(1.0, 1.0)
            lam = max(lam, 1-lam) # to ensure lam >= 0.5

            mixed_im = lam * im + (1 - lam) * im2
            mixed_label = lam * label_a_onehot + (1 - lam) * label_b_onehot

            return mixed_im, mixed_label
        else:
            return im, label_a_onehot


# Resnet
class Residual_Network(nn.Module):
    def __init__(self):
        super(Residual_Network, self).__init__()
        
        self.cnn_layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
        )

        self.cnn_layer2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
        )

        self.cnn_layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
        )

        self.cnn_layer4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
        )
        self.cnn_layer5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
        )
        self.cnn_layer6 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
        )
        self.fc_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256* 1 * 1, 128),
            nn.ReLU(),
            nn.Linear(128, 11)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # input (x): [batch_size, 3, 128, 128]
        # output: [batch_size, 11]

        # Extract features by convolutional layers.
        x1 = self.cnn_layer1(x)
        
        x1 = self.relu(x1)
        
        x2 = self.cnn_layer2(x1)
        
        x2 = x1 + x2

        x2 = self.relu(x2)
        
        x3 = self.cnn_layer3(x2)
        
        x3 = self.relu(x3)
        
        x4 = self.cnn_layer4(x3)
        
        x4 = x3 + x4

        x4 = self.relu(x4)
        
        x5 = self.cnn_layer5(x4)
        
        x5 = self.relu(x5)
        
        x6 = self.cnn_layer6(x5)
        
        x6 = x5 + x6

        x6 = self.relu(x6)
        
        # The extracted feature map must be flatten before going to fully-connected layers.
        # The features are transformed by fully-connected layers to obtain the final logits.
        xout = self.fc_layer(x6)
        return xout
    
# Config
batch_size = 64
_dataset_dir = "./food11"

device = "cuda" if torch.cuda.is_available() else "cpu"

n_epochs = 200
patience = n_epochs // 10
k_folds = 4

model = Residual_Network().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5) 
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-7)

train_dir = os.path.join(_dataset_dir, "training")
valid_dir = os.path.join(_dataset_dir, "validation")

train_files = [os.path.join(train_dir, x) for x in os.listdir(train_dir) if x.endswith(".jpg")]
valid_files = [os.path.join(valid_dir, x) for x in os.listdir(valid_dir) if x.endswith(".jpg")]

all_files = np.array(sorted(train_files + valid_files))

kfold = KFold(n_splits=k_folds, shuffle=True, random_state=myseed)

for fold, (train_idx, valid_idx) in enumerate(kfold.split(all_files)):
    print(f"Fold {fold+1}/{k_folds}")
    train_files_fold = all_files[train_idx]
    valid_files_fold = all_files[valid_idx]

    train_dataset = FoodDataset(path=None, tfm=train_tfm, files=train_files_fold, mixup=False)
    valid_dataset = FoodDataset(path=None, tfm=test_tfm, files=valid_files_fold, mixup=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    model = Residual_Network().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5) 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-7)

    best_valid_acc = 0.0
    stale = 0
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} - Training"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            if labels.dim() > 1:
                _, labels_max = torch.max(labels.data, 1)
            else:
                labels_max = labels
            total_train += labels.size(0)
            correct_train += (predicted == labels_max).sum().item()

        scheduler.step()

        train_acc = correct_train / total_train
        print(f"Fold {fold+1}, Epoch {epoch+1}, Train Loss: {train_loss/total_train:.4f}, Train Acc: {train_acc:.4f}")

        model.eval()
        valid_loss = 0.0
        correct_valid = 0
        total_valid = 0

        with torch.no_grad():
            for images, labels in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{n_epochs} - Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                valid_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                if labels.dim() > 1:
                    _, labels_max = torch.max(labels.data, 1)
                else:
                    labels_max = labels
                total_valid += labels.size(0)
                correct_valid += (predicted == labels_max).sum().item()

        valid_acc = correct_valid / total_valid
        print(f"Fold {fold+1}, Epoch {epoch+1}, Valid Loss: {valid_loss/total_valid:.4f}, Valid Acc: {valid_acc:.4f}")

        # update logs
        if valid_acc > best_valid_acc:
            with open(f"./{_exp_name}_log.txt","a"):
                print(f"[ Fold {fold+1} | Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss/total_valid:.5f}, acc = {valid_acc:.5f} -> best")
        else:
            with open(f"./{_exp_name}_log.txt","a"):
                print(f"[ Fold {fold+1} | Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss/total_valid:.5f}, acc = {valid_acc:.5f}")


        # save model
        if valid_acc > best_valid_acc:
            print(f"Best model found at epoch {epoch}, saving model")
            torch.save(model.state_dict(), f"{_exp_name}_fold{fold+1}_best.ckpt") # only save best to prevent output memory exceed error
            best_valid_acc = valid_acc
            stale = 0
        else:
            stale += 1
            if stale > patience:
                print(f"No improvment {patience} consecutive epochs, early stopping")
                break


