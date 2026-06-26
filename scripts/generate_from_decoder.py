import torch
import torch.nn as nn
import argparse
import os
import matplotlib.pyplot as plt


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


OUTPUT_DIR = "hasil"

def main():
    parser = argparse.ArgumentParser(description="Generasi citra Fashion-MNIST dari Decoder")
    parser.add_argument("--decoder", type=str, required=True, help="Path ke file decoder .pth")
    parser.add_argument("--z1", type=float, help="Nilai pertama latent vector (untuk dim=2)")
    parser.add_argument("--z2", type=float, help="Nilai kedua latent vector (untuk dim=2)")
    parser.add_argument("--latent", type=str, help="Latent vector sebagai string (contoh: '0.5,-1.2,0.3,0.8')")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Direktori output gambar")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(args.decoder, map_location=device)
    if "model.0.weight" in state:
        latent_dim = state["model.0.weight"].shape[1]
    else:
        raise ValueError("Tidak dapat menentukan latent_dim dari state dict")

    decoder = Decoder(latent_dim).to(device)
    decoder.load_state_dict(state)
    decoder.eval()

    if args.latent:
        values = [float(x.strip()) for x in args.latent.split(",")]
        if len(values) != latent_dim:
            raise ValueError(f"Latent vector harus memiliki {latent_dim} nilai, tapi mendapat {len(values)}")
        z = torch.tensor(values, dtype=torch.float32).unsqueeze(0)
    elif args.z1 is not None and args.z2 is not None:
        if latent_dim != 2:
            raise ValueError(f"--z1/--z2 hanya untuk latent_dim=2, model ini latent_dim={latent_dim}")
        z = torch.tensor([[args.z1, args.z2]], dtype=torch.float32)
    else:
        parser.print_help()
        print("\nGunakan --latent atau --z1 --z2 untuk memberikan latent vector")
        return

    z = z.to(device)
    with torch.no_grad():
        output = decoder(z)
    img = output.view(28, 28).cpu()

    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.savefig(f"{args.output_dir}/generated_image.png", bbox_inches="tight")
    print(f"Gambar disimpan: {args.output_dir}/generated_image.png")


if __name__ == "__main__":
    main()
