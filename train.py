"""Training script for AE and VAE models."""

import argparse
import pathlib
from typing import Tuple

import tensorflow as tf

from models import Autoencoder, VariationalAutoencoder
from utils.dataloader import build_image_datasets
from utils.viz import plot_reconstructions, plot_training_history

DATA_DIR_DEFAULT = "./medical-mnist"
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 128
EPOCHS_AE = 20
EPOCHS_VAE = 20
LATENT_DIM_FULL = 16
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.15
TEST_SPLIT = 0.10
CLASS_NAMES = [
    "AbdomenCT",
    "BreastMRI",
    "ChestCT",
    "CXR",
    "Hand",
    "HeadCT",
]


def train_autoencoder(train_ds, val_ds) -> tf.keras.callbacks.History:
    """Compile and train the autoencoder model."""
    autoencoder = Autoencoder(latent_dim=LATENT_DIM_FULL, input_shape=(*IMAGE_SIZE, 1))
    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.MeanSquaredError(),
    )

    history = autoencoder.fit(
        train_ds,
        epochs=EPOCHS_AE,
        validation_data=val_ds,
        verbose=1,
    )
    return autoencoder, history


def train_variational_autoencoder(train_ds, val_ds) -> Tuple[VariationalAutoencoder, dict]:
    """Compile and train the variational autoencoder."""
    vae = VariationalAutoencoder(latent_dim=LATENT_DIM_FULL, input_shape=(*IMAGE_SIZE, 1))
    vae.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        run_eagerly=False,
    )

    history = vae.fit(
        train_ds,
        epochs=EPOCHS_VAE,
        validation_data=val_ds,
        verbose=1,
    )
    return vae, history


def save_model_weights(model: tf.keras.Model, save_dir: pathlib.Path, filename: str) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_weights(save_dir / filename)


def main(data_dir: str = DATA_DIR_DEFAULT) -> dict:
    """Main entry point for training AE and VAE models."""
    data_dir_path = pathlib.Path(data_dir)
    if not data_dir_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    train_ds, val_ds, test_ds, label_ds = build_image_datasets(
        root_dir=str(data_dir_path),
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        val_split=VAL_SPLIT,
        test_split=TEST_SPLIT,
        class_names=CLASS_NAMES,
    )

    autoencoder, ae_history = train_autoencoder(train_ds, val_ds)
    plot_training_history(ae_history)
    plot_reconstructions(autoencoder, test_ds.take(1), num_images=8)
    save_model_weights(autoencoder, pathlib.Path("saved_models"), "ae_weights.h5")

    vae, vae_history = train_variational_autoencoder(train_ds, val_ds)
    plot_training_history(vae_history)
    plot_reconstructions(vae, test_ds.take(1), num_images=8)
    save_model_weights(vae, pathlib.Path("saved_models"), "vae_weights.h5")

    return {
        "ae_history": ae_history,
        "vae_history": vae_history,
        "train_dataset": train_ds,
        "val_dataset": val_ds,
        "test_dataset": test_ds,
        "label_dataset": label_ds,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AE and VAE models on image data.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DATA_DIR_DEFAULT,
        help="Path to the dataset root directory containing image folders.",
    )
    args = parser.parse_args()
    main(args.data_dir)
