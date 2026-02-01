import torch
import torch.nn.functional as F
from model import VAE
from dataset import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def vae_loss(recon_x, x, mu, logvar):
    recon = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl

def train():
    train_loader, _ = get_dataloaders()
    model = VAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 10

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x, _ in train_loader:
            x = x.view(-1, 28*28).to(device)

            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss = vae_loss(recon, x, mu, logvar)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader.dataset):.4f}")

    torch.save(model.state_dict(), "vae.pth")

if __name__ == "__main__":
    train()
