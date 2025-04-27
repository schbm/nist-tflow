import pytest
import tensorflow as tf
import numpy as np
from data import (
    preprocess_ndarray,
    preprocess_dataset,
    normalize_convert_images_f32,
    convert_label_onehot,
)


# Mocking TensorFlow's data loading functions
@pytest.fixture
def mock_mnist():
    train_images = np.random.randint(0, 255, (60000, 28, 28), dtype=np.uint8)
    train_labels = np.random.randint(0, 10, 60000, dtype=np.int64)
    test_images = np.random.randint(0, 255, (10000, 28, 28), dtype=np.uint8)
    test_labels = np.random.randint(0, 10, 10000, dtype=np.int64)
    return (train_images, train_labels), (test_images, test_labels)


def test_preprocess_ndarray(mock_mnist):
    (train_images, train_labels), (test_images, test_labels) = mock_mnist

    train_dataset = preprocess_ndarray(train_images, train_labels)
    test_dataset = preprocess_ndarray(test_images, test_labels)

    for image, label in train_dataset.take(1):
        assert image.shape == (28, 28)
        assert label.shape == (10,)

    for image, label in test_dataset.take(1):
        assert image.shape == (28, 28)
        assert label.shape == (10,)


def test_preprocess_dataset(mock_mnist):
    (train_images, train_labels), (test_images, test_labels) = mock_mnist
    full_train_dataset = preprocess_ndarray(train_images, train_labels)
    processed_train_dataset = preprocess_dataset(full_train_dataset, batch_size=32)

    # Check if batching works
    for batch in processed_train_dataset.take(1):
        images, labels = batch
        assert images.shape == (32, 28, 28)
        assert labels.shape == (32, 10)


def test_normalize_convert_images_f32():
    image = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
    normalized_image = normalize_convert_images_f32(image)

    assert normalized_image.dtype == tf.float32
    assert np.all(normalized_image >= 0)
    assert np.all(normalized_image <= 1)


def test_convert_label_onehot():
    labels = np.array([0, 1, 2, 3, 4, 5])
    onehot_labels = convert_label_onehot(labels, classes=10)

    assert onehot_labels.shape == (6, 10)
    assert np.sum(onehot_labels[0]) == 1


if __name__ == "__main__":
    print("\n[MAIN] Running pytest...")
    pytest.main(["-v", __file__])
