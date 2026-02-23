import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

# Optional: Download dataset
#
try:
    import kagglehub
    path = kagglehub.dataset_download("soumikrakshit/div2k-high-resolution-images")
    print("Dataset path:", path)
except:
    path = "./data"  
    print("Using local data folder:", path)

# ==========================
# Dataset
# ==========================
class DT(Dataset):
    def __init__(self,root,split):
        self.root=f"{root}/DIV2K_{split}_HR/DIV2K_{split}_HR"
        self.hr_trans=T.Compose([
            T.RandomCrop((256,256)),
            T.ToTensor()
        ])
        self.lr_trans=T.Compose([
            T.Resize((64,64)),
            T.Resize((256,256),interpolation=T.InterpolationMode.BICUBIC)
        ])
        self.samples=os.listdir(self.root)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,idx):
        img_name=self.samples[idx]
        img_path=os.path.join(self.root,img_name)
        image=Image.open(img_path).convert("RGB")
        hr_img=self.hr_trans(image)
        lr_img=self.lr_trans(hr_img)
        return hr_img, lr_img

# ==========================
# SRCNN Model
# ==========================
class SRCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,64,9,1,4)
        self.batch1=nn.BatchNorm2d(64)
        self.conv2=nn.Conv2d(64,32,1,1,0)
        self.batch2=nn.BatchNorm2d(32)
        self.conv3=nn.Conv2d(32,3,5,1,2)

    def forward(self,x):
        res=x
        out=F.relu(self.batch1(self.conv1(x)))
        out=F.relu(self.batch2(self.conv2(out)))
        out=self.conv3(out)
        return out+res

# ==========================
# Training & Testing functions
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
train_data=DT(path,'train')
valid_data=DT(path,'valid')

train_loader=DataLoader(train_data,batch_size=32,shuffle=True,num_workers=2)
test_loader=DataLoader(valid_data,batch_size=32,shuffle=False,num_workers=2)

# Init model, loss, optimizer
model = SRCNN().to(device)
lossfun = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def training(model,train_loader,optimizer,lossfun):
    losses=[]
    model.train()
    for high,low in train_loader:
        high,low=high.to(device),low.to(device)
        optimizer.zero_grad()
        yhat=model(low)
        loss=lossfun(yhat,high)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)

def testing(model,test_loader,lossfun):
    losses=[]
    model.eval()
    with torch.no_grad():
        for high,low in test_loader:
            high,low=high.to(device),low.to(device)
            yhat=model(low)
            loss=lossfun(yhat,high)
            losses.append(loss.item())
    return np.mean(losses)

# ==========================
# Main Training Loop
# ==========================
losses={'train_loss':[],'test_loss':[]}

for epoch in range(50):
    start=time.time()
    train_loss=training(model,train_loader,optimizer,lossfun)
    test_loss=testing(model,test_loader,lossfun)
    epoch_time = time.time()-start

    print(f'Epoch {epoch+1} --- train_loss {train_loss:.4f} --- test_loss {test_loss:.4f} --- time {epoch_time:.2f} sec')
    losses['train_loss'].append(train_loss)
    losses['test_loss'].append(test_loss)
