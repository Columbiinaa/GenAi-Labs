import torch
import matplotlib.pyplot as plt
from model import VAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VAE()
model.load_state_dict(torch.load("vae.pth"))
model.to(device)
model.eval()

with torch.no_grad():
    z = torch.randn(16, 20).to(device)
    samples = model.decode(z).cpu()

plt.figure(figsize=(4,4))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.imshow(samples[i].view(28,28), cmap='gray')
    plt.axis('off')

plt.show()
