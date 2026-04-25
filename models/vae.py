"""Variational autoencoder models."""

from typing import Optional, Tuple

import tensorflow as tf
from utils.losses import kl_divergence_loss, reconstruction_loss


class VAEEncoder(tf.keras.Model):
    """Encoder that produces z_mean and z_log_var."""

    def __init__(self, latent_dim: int, input_shape=(64, 64, 1)) -> None:
        super().__init__(name="vae_encoder")
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, activation="relu", padding="same"),
            tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, activation="relu", padding="same"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(latent_dim * 2),
        ], name="vae_encoder_stack")

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return self.encoder(inputs)


class VAEDecoder(tf.keras.Model):
    """Decoder that maps latent vectors back to image logits."""

    def __init__(self, latent_dim: int, input_shape=(64, 64, 1)) -> None:
        super().__init__(name="vae_decoder")
        spatial_dim = input_shape[0] // 4
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(latent_dim,)),
            tf.keras.layers.Dense(spatial_dim * spatial_dim * 32, activation="relu"),
            tf.keras.layers.Reshape((spatial_dim, spatial_dim, 32)),
            tf.keras.layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding="same", activation="relu"),
            tf.keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding="same", activation="relu"),
            tf.keras.layers.Conv2DTranspose(input_shape[2], kernel_size=3, strides=1, padding="same"),
        ], name="vae_decoder_stack")

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return self.decoder(inputs)


class VariationalAutoencoder(tf.keras.Model):
    """Variational Autoencoder with a custom training loop."""

    def __init__(self, latent_dim: int = 2, input_shape=(64, 64, 1)) -> None:
        super().__init__(name="variational_autoencoder")
        self.latent_dim = latent_dim
        self.encoder = VAEEncoder(latent_dim, input_shape=input_shape)
        self.decoder = VAEDecoder(latent_dim, input_shape=input_shape)

    def encode(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        mean_logvar = self.encoder(inputs)
        mean, log_var = tf.split(mean_logvar, num_or_size_splits=2, axis=1)
        return mean, log_var

    def reparameterize(self, mean: tf.Tensor, log_var: tf.Tensor) -> tf.Tensor:
        eps = tf.random.normal(shape=tf.shape(mean))
        return mean + tf.exp(0.5 * log_var) * eps

    def decode(self, z: tf.Tensor, apply_sigmoid: bool = False) -> tf.Tensor:
        logits = self.decoder(z)
        if apply_sigmoid:
            return tf.sigmoid(logits)
        return logits

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        mean, log_var = self.encode(inputs)
        z = self.reparameterize(mean, log_var)
        return self.decode(z, apply_sigmoid=True)

    def train_step(self, data: tf.Tensor) -> dict:
        inputs = data
        with tf.GradientTape() as tape:
            mean, log_var = self.encode(inputs)
            z = self.reparameterize(mean, log_var)
            logits = self.decoder(z)
            recon_loss = reconstruction_loss(inputs, logits)
            kl_loss = kl_divergence_loss(mean, log_var)
            total_loss = recon_loss + kl_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {
            "loss": total_loss,
            "reconstruction_loss": recon_loss,
            "kl_loss": kl_loss,
        }

    def test_step(self, data: tf.Tensor) -> dict:
        inputs = data
        mean, log_var = self.encode(inputs)
        z = self.reparameterize(mean, log_var)
        logits = self.decoder(z)
        recon_loss = reconstruction_loss(inputs, logits)
        kl_loss = kl_divergence_loss(mean, log_var)
        total_loss = recon_loss + kl_loss

        return {
            "loss": total_loss,
            "reconstruction_loss": recon_loss,
            "kl_loss": kl_loss,
        }

    def sample(self, eps: Optional[tf.Tensor] = None, num_samples: int = 16) -> tf.Tensor:
        if eps is None:
            eps = tf.random.normal(shape=(num_samples, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)
