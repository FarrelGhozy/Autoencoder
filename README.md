# Autoencoder Fashion-MNIST

**Farrel Ghozy Affifudin — 452024611053**  
Teknik Informatika — Universitas Darussalam Gontor

## Deskripsi
Proyek ini melatih autoencoder untuk merekonstruksi citra Fashion-MNIST menggunakan PyTorch dengan 3 ukuran latent dimension (2, 8, 32).

## Persyaratan
- Python 3.8+
- PyTorch
- torchvision
- matplotlib
- argparse

## Cara Menjalankan Training di Kaggle
1. Buka [Kaggle](https://kaggle.com) dan buat notebook baru
2. Upload `notebook/autoencoder_fashion_mnist.ipynb`
3. Set akselerator ke **GPU T4 x2**
4. Jalankan semua cell
5. Download file `.pth` dan letakkan di folder `models/`

## Cara Menjalankan Rekonstruksi dari Terminal

### Opsi A: Autoencoder Penuh
```bash
python scripts/reconstruct.py --model models/autoencoder_dim2.pth --index 10
```
Output: `hasil/original.png`, `hasil/reconstructed.png`, `hasil/comparison.png`

### Opsi B: Decoder Saja (Latent Dim = 2)
```bash
python scripts/generate_from_decoder.py --decoder models/decoder_dim2.pth --z1 0.5 --z2 -1.2
```
Output: `hasil/generated_image.png`

### Opsi B: Decoder Saja (Latent Dim > 2)
```bash
python scripts/generate_from_decoder.py --decoder models/decoder_dim8.pth --latent "0.5,-1.2,0.3,0.8,0.1,-0.5,0.7,0.2"
```
Output: `hasil/generated_image.png`

## Struktur Folder
```
FarrelGhozy_452024611053_autoencoder/
├── laporan/          # Laporan LaTeX + PDF
│   ├── laporan.tex
│   └── laporan.pdf
├── notebook/         # Notebook Kaggle
│   └── autoencoder_fashion_mnist.ipynb
├── models/           # Model hasil training (.pth)
│   ├── autoencoder_dim2.pth
│   ├── autoencoder_dim8.pth
│   ├── autoencoder_dim32.pth
│   └── decoder_dim*.pth
├── scripts/          # Program terminal
│   ├── reconstruct.py
│   └── generate_from_decoder.py
├── hasil/            # Output gambar
│   ├── original.png
│   ├── reconstructed.png
│   ├── comparison.png
│   └── generated_image.png
├── README.md
└── .gitignore
```
