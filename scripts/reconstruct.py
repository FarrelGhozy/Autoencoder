import torch
import torch.nn as nn
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent)


OUTPUT_DIR = "hasil"

def main():
    parser = argparse.ArgumentParser(description="Rekonstruksi Fashion-MNIST dengan Autoencoder")
    parser.add_argument("--model", type=str, required=True, help="Path ke file model .pth")
    parser.add_argument("--index", type=int, required=True, help="Indeks data Fashion-MNIST")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Direktori output gambar")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    latent_dim = None
    state = torch.load(args.model, map_location=device)
    if "encoder.model.4.weight" in state:
        latent_dim = state["encoder.model.4.weight"].shape[0]
    else:
        raise ValueError("Tidak dapat menentukan latent_dim dari state dict")

    model = Autoencoder(latent_dim).to(device)
    model.load_state_dict(state)
    model.eval()

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)

    img, label = dataset[args.index]
    img_flat = img.view(1, -1).to(device)

    with torch.no_grad():
        output_flat = model(img_flat)
    output_img = output_flat.view(1, 28, 28).cpu()

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    classes = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
    ]

    axes[0].imshow(img.squeeze(), cmap="gray")
    axes[0].set_title(f"Asli: {classes[label]}")
    axes[0].axis("off")
    plt.savefig(f"{args.output_dir}/original.png", bbox_inches="tight")

    axes[1].imshow(output_img.squeeze(), cmap="gray")
    axes[1].set_title("Rekonstruksi")
    axes[1].axis("off")
    plt.savefig(f"{args.output_dir}/reconstructed.png", bbox_inches="tight")

    axes[2].imshow(img.squeeze(), cmap="gray")
    axes[2].imshow(output_img.squeeze(), cmap="gray", alpha=0.5)
    axes[2].set_title("Overlay")
    axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/comparison.png", bbox_inches="tight")
    print(f"Gambar disimpan: {args.output_dir}/original.png, {args.output_dir}/reconstructed.png, {args.output_dir}/comparison.png")


if __name__ == "__main__":
    main()
