"""Loss utilities for AE and VAE models."""

import tensorflow as tf


def reconstruction_loss(x: tf.Tensor, x_logits: tf.Tensor) -> tf.Tensor:
    """Compute binary cross-entropy reconstruction loss from logits."""
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
    return loss_fn(x, x_logits)


def kl_divergence_loss(z_mean: tf.Tensor, z_log_var: tf.Tensor) -> tf.Tensor:
    """Compute the KL divergence for a Gaussian latent distribution."""
    kl_terms = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    kl = -0.5 * tf.reduce_sum(kl_terms, axis=1)
    return tf.reduce_mean(kl)


def vae_elbo_loss(x: tf.Tensor, x_logits: tf.Tensor, z_mean: tf.Tensor, z_log_var: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Return total VAE ELBO loss, reconstruction loss, and KL loss."""
    recon_loss = reconstruction_loss(x, x_logits)
    kl_loss = kl_divergence_loss(z_mean, z_log_var)
    return recon_loss + kl_loss, recon_loss, kl_loss
