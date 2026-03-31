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

class FoodDataset(Dataset):

    def __init__(self,path,tfm=test_tfm,files = None, mixup = False, classes = 11):
        super(FoodDataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".jpg")])
        if files != None:
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
    
_dataset_dir = "./food11"
test_dir = os.path.join(_dataset_dir, "test")
batch_size = 64
device = "cuda" if torch.cuda.is_available() else "cpu"


test_set = FoodDataset(test_dir, tfm=test_tfm, mixup=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

TAA_test_set = FoodDataset(test_dir, tfm=train_tfm, mixup=False)
TAA_test_loader = DataLoader(TAA_test_set, batch_size=batch_size, shuffle=False, num_workers=4)

models = []

for i in range (0, 4):
    model = Residual_Network().to(device)
    model.load_state_dict(torch.load(f"./{_exp_name}_fold{i+1}_best.ckpt"))
    model.eval()
    models.append(model)

test_preds = []
with torch.no_grad():
    for images, _ in tqdm(test_loader):
        images = images.to(device)
        batch_preds = []
        for model in models:
            preds = model(images)
            batch_preds.append(preds.cpu().numpy())
        batch_preds = np.mean(batch_preds, axis=0)
        test_preds.append(batch_preds)

TAA_test_preds = []
with torch.no_grad():
    for images, _ in tqdm(TAA_test_loader):
        images = images.to(device)
        batch_preds = []
        for model in models:
            preds = model(images)
            batch_preds.append(preds.cpu().numpy())
        batch_preds = np.mean(batch_preds, axis=0)
        TAA_test_preds.append(batch_preds)


test_preds = np.concatenate(test_preds, axis=0)
TAA_test_preds = np.concatenate(TAA_test_preds, axis=0)

# print("test_preds_all shape:", test_preds.shape) 
# print("TAA_test_preds_all shape:", TAA_test_preds.shape)

# result = test_preds * 0.8 + TAA_test_preds * 0.2
# print("result shape:", result.shape)


result = []
result = test_preds*0.8 + TAA_test_preds*0.2
result = np.argmax(result, axis=1)

#create test csv
def pad4(i):
    return "0"*(4-len(str(i)))+str(i)
df = pd.DataFrame()
df["Id"] = [pad4(i) for i in range(1,len(test_set)+1)]
df["Category"] = result
df.to_csv("submission.csv",index = False)
