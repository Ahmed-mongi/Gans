"""Visualization helpers for model outputs and training history."""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def plot_reconstructions(model, dataset, num_images: int = 8) -> None:
    """Plot original images and model reconstructions."""
    batch = next(iter(dataset))
    original_images = batch[:num_images]
    reconstructed = model(original_images)

    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4))
    fig.suptitle("Reconstruction Results", fontsize=14)

    for i in range(num_images):
        axes[0, i].imshow(original_images[i].numpy().squeeze(), cmap="gray")
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_ylabel("Original", fontsize=10)

        axes[1, i].imshow(reconstructed[i].numpy().squeeze(), cmap="gray")
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_ylabel("Reconstructed", fontsize=10)

    plt.tight_layout()
    plt.show()


def plot_training_history(history: dict) -> None:
    """Plot training and validation loss curves."""
    if hasattr(history, "history"):
        history = history.history

    plt.figure(figsize=(10, 4))
    plt.plot(history.get("loss", []), label="Train Loss")
    plt.plot(history.get("val_loss", []), label="Val Loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    if "reconstruction_loss" in history or "kl_loss" in history:
        plt.figure(figsize=(10, 4))
        if "reconstruction_loss" in history:
            plt.plot(history.get("reconstruction_loss", []), label="Train Reconstruction")
            plt.plot(history.get("val_reconstruction_loss", []), label="Val Reconstruction")
        if "kl_loss" in history:
            plt.plot(history.get("kl_loss", []), label="Train KL")
            plt.plot(history.get("val_kl_loss", []), label="Val KL")
        plt.title("VAE Loss Components")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.show()


def plot_latent_space(model, dataset, class_names, num_points: int = 2000) -> None:
    """Plot a 2-D latent space if the model exposes an encoder."""
    images, labels = [], []
    for batch_images, batch_labels in dataset:
        if len(images) * batch_images.shape[0] >= num_points:
            break
        images.append(batch_images.numpy())
        labels.append(batch_labels.numpy())

    if not images:
        raise ValueError("Dataset must yield (image, label) pairs for latent visualization.")

    images = np.concatenate(images, axis=0)[:num_points]
    labels = np.concatenate(labels, axis=0)[:num_points]

    if not hasattr(model, "encode"):
        raise AttributeError("Model must implement encode() for latent space plotting.")

    mean, _ = model.encode(tf.convert_to_tensor(images))
    z = mean.numpy()

    plt.figure(figsize=(8, 6))
    for idx, class_name in enumerate(class_names):
        mask = labels == idx
        plt.scatter(z[mask, 0], z[mask, 1], s=6, alpha=0.7, label=class_name)

    plt.title("2D Latent Space")
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.legend(markerscale=2, fontsize=8)
    plt.grid(True)
    plt.show()
