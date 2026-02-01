import torch
import matplotlib.pyplot as plt
from model import VAE
from dataset import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_, test_loader = get_dataloaders()

model = VAE()
model.load_state_dict(torch.load("vae.pth"))
model.to(device)
model.eval()

x, _ = next(iter(test_loader))
x = x.view(-1, 28*28).to(device)

with torch.no_grad():
    recon, _, _ = model(x)

plt.figure(figsize=(6,3))
for i in range(10):
    plt.subplot(2,10,i+1)
    plt.imshow(x[i].view(28,28).cpu(), cmap='gray')
    plt.axis('off')

    plt.subplot(2,10,i+11)
    plt.imshow(recon[i].view(28,28).cpu(), cmap='gray')
    plt.axis('off')

plt.show()
