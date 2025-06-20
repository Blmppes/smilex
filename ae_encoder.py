import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split


def build_autoencoder(input_dim: int, latent_dim: int):
    # Encoder
    encoder_input = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(latent_dim, activation='relu')(encoder_input)
    encoder = models.Model(encoder_input, encoded, name='encoder')

    # Decoder
    decoder_input = layers.Input(shape=(latent_dim,))
    decoded = layers.Dense(input_dim, activation='relu')(decoder_input)
    decoder = models.Model(decoder_input, decoded, name='decoder')

    # Full AE
    ae_input = layers.Input(shape=(input_dim,))
    encoded_repr = encoder(ae_input)
    reconstructed = decoder(encoded_repr)
    autoencoder = models.Model(ae_input, reconstructed)

    return autoencoder, encoder


def run_ae_encoding(input_csv, output_file, latent_dim=16, epochs=10):
    print(f"Loading and preparing data from {input_csv}...")
    data = np.loadtxt(input_csv, delimiter=",", skiprows=1)  # assumes numerical CSV

    x_train, x_test = train_test_split(data, test_size=0.3, random_state=1)
    input_dim = data.shape[1]

    autoencoder, encoder = build_autoencoder(input_dim=input_dim, latent_dim=latent_dim)
    autoencoder.compile(optimizer='adam', loss='mse')

    print("Training Autoencoder...")
    autoencoder.fit(x_train, x_train, epochs=epochs, batch_size=32, verbose=1)

    print("Encoding full data...")
    latent = encoder.predict(data)
    np.save(output_file, latent)
    print(f"Saved AE-encoded features to {output_file}.npy")

    encoder.save(f"{output_file}_encoder.h5")
    print(f"Saved encoder model to {output_file}_encoder.h5")
