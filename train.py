# train.py

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm  
from model import UNet
import lpips

class WatermarkDataset(Dataset):
    def __init__(self, root):
        self.watermarked = sorted(os.listdir(os.path.join(root, "watermark")))
        self.clean = sorted(os.listdir(os.path.join(root, "no-watermark")))
        self.masks = sorted(os.listdir(os.path.join(root, "masks")))
        self.root = root


        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.watermarked)

    def __getitem__(self, idx):
        w_img = Image.open(os.path.join(self.root, "watermark", self.watermarked[idx])).convert("RGB")
        c_img = Image.open(os.path.join(self.root, "no-watermark", self.clean[idx])).convert("RGB")
        m_img = Image.open(os.path.join(self.root, "masks", self.masks[idx])).convert("L")

        w_img = self.transform(w_img)
        c_img = self.transform(c_img)
        m_img = self.transform(m_img)

        input_img = torch.cat([w_img, m_img], dim=0)  
        return input_img, c_img



def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    mse_loss = torch.nn.MSELoss()
    lpips_loss = lpips.LPIPS(net='vgg').to(device)  

    dataset = WatermarkDataset("train")
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)

    num_epochs = 50
    for epoch in range(num_epochs):
        epoch_loss = 0
        for input_img, clean_img in tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_img = input_img.to(device)
            clean_img = clean_img.to(device)

            out = model(input_img)
            
           
            loss = 0.7 * mse_loss(out, clean_img) + 0.3 * lpips_loss(out, clean_img).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1} Average Loss: {epoch_loss / len(loader):.4f}")

    torch.save(model.state_dict(), "watermark_remover.pth")
    


if __name__ == "__main__":
    train()
