import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np

# ==========================
# Dataset & DataLoader
# ==========================
class Constructor:
    def __init__(self, batch_size=64):
        self.transformer = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

        self.train_dataset = datasets.MNIST(
            root="./data", train=True, download=True, transform=self.transformer
        )
        self.test_dataset = datasets.MNIST(
            root="./data", train=False, download=True, transform=self.transformer
        )

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False
        )

# ==========================
# Autoencoder Model
# ==========================
class AETW(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(784, 256)
        self.encoder_layer = nn.Linear(256, 50)
        self.decoder_layer = nn.Linear(50, 256)
        self.output_layer = nn.Linear(256, 784)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.encoder_layer(x))
        x = F.relu(self.decoder_layer(x))
        x = torch.tanh(self.output_layer(x))
        return x

# ==========================
# Trainer
# ==========================
class Trainor:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.lossfun = nn.MSELoss()
        self.losses = {"train_loss": [], "test_loss": []}

    def train_model(self, train_loader, test_loader=None, epochs=10, device="cpu"):
        self.model.to(device)

        for epoch in range(epochs):
            train_batch_loss = []

            for images, _ in train_loader:
                images = images.view(images.size(0), -1).to(device)
                yhat = self.model(images)
                loss = self.lossfun(yhat, images)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_batch_loss.append(loss.item())

            self.losses["train_loss"].append(np.mean(train_batch_loss))

            if test_loader is not None:
                test_batch_loss = []
                with torch.no_grad():
                    for images, _ in test_loader:
                        images = images.view(images.size(0), -1).to(device)
                        yhat = self.model(images)
                        loss = self.lossfun(yhat, images)
                        test_batch_loss.append(loss.item())
                    self.losses["test_loss"].append(np.mean(test_batch_loss))
                    print(
                        f"Epoch {epoch+1}/{epochs} - Train Loss: {self.losses['train_loss'][-1]:.4f} | Test Loss: {self.losses['test_loss'][-1]:.4f}"
                    )
            else:
                print(
                    f"Epoch {epoch+1}/{epochs} - Train Loss: {self.losses['train_loss'][-1]:.4f}"
                )

# ==========================
# Main
# ==========================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    data = Constructor(batch_size=128)
    train_loader = data.train_loader
    test_loader = data.test_loader  # Optional

    # Init Model & Optimizer
    model = AETW()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train
    trainer = Trainor(model=model, optimizer=optimizer)
    trainer.train_model(train_loader=train_loader, test_loader=None, epochs=300, device=device)
