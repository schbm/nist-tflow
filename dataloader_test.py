import tensorflow as tf
import pytest

from data import DataLoaderMNIST


@pytest.fixture(scope="module")
def mnist_loader():
    return DataLoaderMNIST(
        validation_dataset_size=5000, mini_batch_size=32, debug=False
    )


def test_train_dataset_shapes_and_types(mnist_loader):
    train_dataset = mnist_loader.train_dataset

    for images, labels in train_dataset.take(1):
        assert images.shape == (32, 28, 28)
        assert labels.shape == (32, 10)
        assert images.dtype == tf.float32
        assert labels.dtype == tf.float32
        assert tf.reduce_min(images) >= 0.0
        assert tf.reduce_max(images) <= 1.0


def test_validation_dataset_shapes_and_types(mnist_loader):
    valid_dataset = mnist_loader.valid_dataset

    for images, labels in valid_dataset.take(1):
        assert images.shape == (32, 28, 28)
        assert labels.shape == (32, 10)
        assert images.dtype == tf.float32
        assert labels.dtype == tf.float32
        assert tf.reduce_min(images) >= 0.0
        assert tf.reduce_max(images) <= 1.0


def test_test_dataset_shapes_and_types(mnist_loader):
    test_dataset = mnist_loader.test_dataset

    for images, labels in test_dataset.take(1):
        assert images.shape == (32, 28, 28)
        assert labels.shape == (32, 10)
        assert images.dtype == tf.float32
        assert labels.dtype == tf.float32
        assert tf.reduce_min(images) >= 0.0
        assert tf.reduce_max(images) <= 1.0


if __name__ == "__main__":
    print("\n[MAIN] Running pytest...")
    pytest.main(["-v", __file__])
