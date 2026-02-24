import kagglehub

# Download latest version
path = kagglehub.dataset_download("soumikrakshit/div2k-high-resolution-images")

print("Path to dataset files:", path)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import torchvision.transforms as T


class DT(Dataset):
    def __init__(self,root,split):
        self.root=f"{root}/DIV2K_{split}_HR/DIV2K_{split}_HR"

        self.crop=T.RandomCrop((256,256))
        self.resize=T.Resize((128,128),interpolation=T.InterpolationMode.BICUBIC)
        self.to_tensor=T.ToTensor()

        self.samples=os.listdir(self.root)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,idx):
        img_name=self.samples[idx]

        img_path=os.path.join(self.root,img_name)

        image=Image.open(img_path).convert("RGB")

        hr_img=self.crop(image)
        lr_img=self.resize(hr_img)

        return self.to_tensor(hr_img),self.to_tensor(lr_img)

class ESPCN(nn.Module):

    def __init__(self,scale_factor=2):
        super().__init__()

        self.scale_factor=scale_factor

        self.conv1=nn.Conv2d(3,64,5,1,2)
        self.conv2=nn.Conv2d(64,32,3,1,1)
        self.conv3=nn.Conv2d(32,(3*self.scale_factor**2),3,1,1)

        self.pixel_shuffle=nn.PixelShuffle(self.scale_factor)


    def forward(self,x):
        out=F.relu(self.conv1(x))
        out=F.relu(self.conv2(out))
        out=self.conv3(out)
        return self.pixel_shuffle(out)

train_data=DT(path,'train')
valid_data=DT(path,'valid')

train_loader=DataLoader(train_data,batch_size=16,shuffle=True,num_workers=2,pin_memory=True)
test_loader=DataLoader(valid_data,batch_size=16,shuffle=False,num_workers=2,pin_memory=True)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=ESPCN().to(device)
lossfun=nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=.001)

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

import time
losses={'train_loss':[],'test_loss':[]}

for epoch in range(10):
    current=time.time()
    train_loss=training(model,train_loader,optimizer,lossfun)
    test_loss=testing(model,test_loader,lossfun)
    epoch_time = time.time()-current

    print(f'{epoch+1} --- train_loss {train_loss:.4f} --- test_loss {test_loss:.4f} --- time {epoch_time:.2f}sec')

    losses['train_loss'].append(train_loss)
    losses['test_loss'].append(test_loss)


idx = np.random.choice(len(test_loader), 1)
hr_img, lr_img = test_loader.dataset[idx[0]]

bench_img = lr_img.unsqueeze(0).to(device)
model.eval()
with torch.no_grad():
    yhat = model(bench_img).squeeze().cpu().clamp(0, 1)

hr_img_display = hr_img.permute(1, 2, 0).cpu().numpy()
lr_img_display = lr_img.permute(1, 2, 0).cpu().numpy()
yhat_display = yhat.permute(1, 2, 0).numpy()

fig, axis = plt.subplots(1, 3, figsize=(18, 6))

axis[0].imshow(lr_img_display)
axis[0].set_title("Input: Low Res (128x128)")

axis[1].imshow(yhat_display)
axis[1].set_title("Model: ESPCN Output (256x256)")

axis[2].imshow(hr_img_display)
axis[2].set_title("Target: High Res Original (256x256)")

plt.show()