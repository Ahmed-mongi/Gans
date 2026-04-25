# Autoencoder and Variational Autoencoder Project

This project refactors the Medical MNIST autoencoder/VAE work into a clean modular Python codebase.

## Folder structure

```
project_root/
├── models/
│   ├── __init__.py
│   ├── ae.py
│   └── vae.py
├── utils/
│   ├── __init__.py
│   ├── dataloader.py
│   ├── losses.py
│   └── viz.py
├── train.py
├── main.ipynb
├── requirements.txt
└── README.md
```

## Dataset setup

1. Place the Medical MNIST image folders under a dataset root directory.
2. The expected structure is a root directory containing class folders such as `AbdomenCT`, `BreastMRI`, `ChestCT`, `CXR`, `Hand`, and `HeadCT`.
3. In `main.ipynb`, update `DATA_DIR` to point to your dataset root.

## How to run

### Locally

```bash
python train.py --data-dir <path_to_medical_mnist>
```

### In Google Colab

1. Mount Google Drive.
2. Set `DATA_DIR` in `main.ipynb` to the folder path in Drive.
3. Run the notebook cells.

## AE and VAE organization

- `models/ae.py`: standard convolutional autoencoder with separate `AEEncoder`, `AEDecoder`, and `Autoencoder` classes.
- `models/vae.py`: variational autoencoder with encoder outputting `z_mean`, `z_log_var`, reparameterisation, decoder, and custom `train_step`/`test_step`.
- `utils/dataloader.py`: tf.data pipeline to load images, resize, normalize, shuffle, batch, and prefetch.
- `utils/losses.py`: reconstruction and KL divergence loss helpers with a correct VAE ELBO formulation.
- `utils/viz.py`: matplotlib helpers to plot reconstructions and training curves.
- `train.py`: main terminal script that builds datasets, trains AE and VAE, saves weights, and displays simple visualizations.

## tf.data usage

The project uses `tf.data` for image loading, decoding, resizing, normalization, shuffling, batching, and prefetching. The pipeline avoids NumPy-based dataset loading for the main flow.
