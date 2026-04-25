"""tf.data dataset utilities for image loading."""

import os
import pathlib
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE
IMAGE_EXTENSIONS = ["*.jpeg", "*.jpg", "*.png"]


def list_image_paths(data_dir: str) -> List[str]:
    """Return a sorted list of image file paths under the given directory."""
    root = pathlib.Path(data_dir)
    paths: List[str] = []
    for pattern in IMAGE_EXTENSIONS:
        paths.extend(sorted(str(path) for path in root.rglob(pattern)))
    return sorted(paths)


def load_and_preprocess_image(
    file_path: tf.Tensor,
    image_size: Tuple[int, int] = (64, 64),
    channels: int = 1,
) -> tf.Tensor:
    """Decode, resize, and normalise a single image tensor."""
    raw = tf.io.read_file(file_path)
    img = tf.image.decode_image(raw, channels=channels, expand_animations=False)
    img = tf.image.resize(img, image_size)
    img = tf.cast(img, tf.float32) / 255.0
    img.set_shape([image_size[0], image_size[1], channels])
    return img


def get_label_from_path(file_path: tf.Tensor, class_names: List[str]) -> tf.Tensor:
    """Extract a class index from the parent folder name."""
    parts = tf.strings.split(file_path, os.sep)
    folder_name = parts[-2]
    equality = [tf.cast(tf.equal(folder_name, name), tf.int32) for name in class_names]
    label = tf.argmax(tf.stack(equality, axis=0), axis=0, output_type=tf.int32)
    return label


def create_image_dataset(
    data_dir: str,
    image_size: Tuple[int, int] = (64, 64),
    batch_size: int = 32,
    shuffle: bool = True,
    buffer_size: int = 1000,
    seed: int = 42,
) -> tf.data.Dataset:
    """Create a tf.data.Dataset of preprocessed images from a folder path."""
    file_paths = list_image_paths(data_dir)
    if not file_paths:
        raise ValueError(f"No image files found in {data_dir}")

    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(buffer_size, len(file_paths)), seed=seed)
    dataset = dataset.map(
        lambda path: load_and_preprocess_image(path, image_size=image_size),
        num_parallel_calls=AUTOTUNE,
    )
    return dataset.batch(batch_size).prefetch(AUTOTUNE)


def create_image_label_dataset(
    data_dir: str,
    class_names: List[str],
    image_size: Tuple[int, int] = (64, 64),
    batch_size: int = 32,
    shuffle: bool = False,
    buffer_size: int = 1000,
    seed: int = 42,
) -> tf.data.Dataset:
    """Create a tf.data.Dataset that yields (image, label) pairs."""
    file_paths = list_image_paths(data_dir)
    if not file_paths:
        raise ValueError(f"No image files found in {data_dir}")

    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(buffer_size, len(file_paths)), seed=seed)
    dataset = dataset.map(
        lambda path: (
            load_and_preprocess_image(path, image_size=image_size),
            get_label_from_path(path, class_names),
        ),
        num_parallel_calls=AUTOTUNE,
    )
    return dataset.batch(batch_size).prefetch(AUTOTUNE)


def build_image_datasets(
    root_dir: str,
    image_size: Tuple[int, int] = (64, 64),
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.10,
    buffer_size: int = 1000,
    seed: int = 42,
    class_names: Optional[List[str]] = None,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Optional[tf.data.Dataset]]:
    """Build train/validation/test datasets using tf.data.

    Returns:
        train_ds, val_ds, test_ds, label_ds
    """
    file_paths = list_image_paths(root_dir)
    if not file_paths:
        raise ValueError(f"No images found in {root_dir}")

    rng = np.random.default_rng(seed)
    file_paths = [file_paths[i] for i in rng.permutation(len(file_paths))]

    num_test = int(len(file_paths) * test_split)
    num_val = int(len(file_paths) * val_split)
    num_train = len(file_paths) - num_val - num_test

    train_paths = file_paths[:num_train]
    val_paths = file_paths[num_train : num_train + num_val]
    test_paths = file_paths[num_train + num_val :]

    def dataset_from_paths(paths: List[str], shuffle: bool) -> tf.data.Dataset:
        dataset = tf.data.Dataset.from_tensor_slices(paths)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=min(buffer_size, len(paths)), seed=seed)
        dataset = dataset.map(
            lambda path: load_and_preprocess_image(path, image_size=image_size),
            num_parallel_calls=AUTOTUNE,
        )
        return dataset.batch(batch_size).prefetch(AUTOTUNE)

    train_ds = dataset_from_paths(train_paths, shuffle=True)
    val_ds = dataset_from_paths(val_paths, shuffle=False)
    test_ds = dataset_from_paths(test_paths, shuffle=False)

    label_ds = None
    if class_names is not None:
        label_ds = tf.data.Dataset.from_tensor_slices(test_paths)
        label_ds = label_ds.map(
            lambda path: (
                load_and_preprocess_image(path, image_size=image_size),
                get_label_from_path(path, class_names),
            ),
            num_parallel_calls=AUTOTUNE,
        )
        label_ds = label_ds.batch(batch_size).prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, label_ds
