"""Standard convolutional autoencoder models."""

import tensorflow as tf


class AEEncoder(tf.keras.Model):
    """Encoder for a convolutional autoencoder."""

    def __init__(self, latent_dim: int, input_shape=(64, 64, 1)) -> None:
        super().__init__(name="ae_encoder")
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, padding="same", activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(latent_dim, activation="relu"),
        ], name="ae_encoder_stack")

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return self.encoder(inputs)


class AEDecoder(tf.keras.Model):
    """Decoder for a convolutional autoencoder."""

    def __init__(self, latent_dim: int, input_shape=(64, 64, 1)) -> None:
        super().__init__(name="ae_decoder")
        spatial_dim = input_shape[0] // 8
        base_filters = 128
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(latent_dim,)),
            tf.keras.layers.Dense(spatial_dim * spatial_dim * base_filters, activation="relu"),
            tf.keras.layers.Reshape((spatial_dim, spatial_dim, base_filters)),
            tf.keras.layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding="same", activation="relu"),
            tf.keras.layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding="same", activation="relu"),
            tf.keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(input_shape[2], kernel_size=3, padding="same", activation="sigmoid"),
        ], name="ae_decoder_stack")

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return self.decoder(inputs)


class Autoencoder(tf.keras.Model):
    """Full autoencoder with separate encoder and decoder."""

    def __init__(self, latent_dim: int = 16, input_shape=(64, 64, 1)) -> None:
        super().__init__(name="autoencoder")
        self.encoder = AEEncoder(latent_dim, input_shape=input_shape)
        self.decoder = AEDecoder(latent_dim, input_shape=input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        encoded = self.encoder(inputs)
        return self.decoder(encoded)
