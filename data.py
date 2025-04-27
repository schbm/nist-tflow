import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# https://www.tensorflow.org/guide/data

print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print("TensorFlow is using the following GPU(s):")
    for gpu in gpus:
        print(gpu)
else:
    print("No GPU detected. TensorFlow will use CPU.")


class DataLoaderMNIST:
    """Provide train, validation, and test datasets of the MNIST dataset."""

    def __init__(self, validation_dataset_size=5000, mini_batch_size=32, debug=True):
        mnist = tf.keras.datasets.mnist

        train, test = mnist.load_data()

        train_images, train_labels = train

        if debug:
            print(
                "#########################################################################################"
            )
            print("printing information about training images and labels")
            print_image_info(train_images)
            print_label_info(train_labels)
            print("saving first image and label as imagename=debug_train_image.png")
            save_debug_image(train_images[0], train_labels[0], "debug_train_image.png")
            print_label_balance(train_labels)
            print(
                "#########################################################################################"
            )

        full_train_dataset = preprocess_ndarray(train_images, train_labels)
        train_dataset, validation_dataset = split_dataset(
            full_train_dataset, validation_size=validation_dataset_size
        )
        train_dataset = preprocess_dataset(train_dataset, batch_size=mini_batch_size)
        validation_dataset = preprocess_dataset(
            validation_dataset, batch_size=mini_batch_size, enable_shuffling=False
        )

        test_images, test_labels = test

        if debug:
            print(
                "#########################################################################################"
            )
            print("printing information about test images and labels")
            print_image_info(test_images)
            print_label_info(test_labels)
            print("saving first image and label as imagename=debug_test_image.png")
            save_debug_image(test_images[0], test_labels[0], "debug_test_image.png")
            print_label_balance(test_labels)
            print(
                "#########################################################################################"
            )

        full_test_dataset = preprocess_ndarray(test_images, test_labels)
        test_dataset = preprocess_dataset(
            full_test_dataset, batch_size=mini_batch_size, enable_shuffling=False
        )

        self._train_dataset = train_dataset  # Use batching and shuffling
        self._valid_dataset = validation_dataset  # Use batching
        self._test_dataset = test_dataset  # Use batching

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def valid_dataset(self):
        return self._valid_dataset

    @property
    def test_dataset(self):
        return self._test_dataset


def batch_dataset(dataset, batch_size):
    """creates a batched dataset
    @type dataset: tf.dataset
    """
    return dataset.batch(batch_size)


def full_shuffle_dataset(dataset):
    """fully randomly shuffles the dataset
    @type dataset: tf.dataset
    """
    return dataset.shuffle(dataset.cardinality())


def normalize_convert_images_f32(images):
    """normalizes between [0,1] and converts to float32
    @type images: numpy.ndarray
    """
    return tf.cast(images, tf.float32) / 255.0


def print_image_info(images):
    """
    @type images: numpy.ndarray
    """
    print(f"images are of type={type(images)}")
    print(f"images have shape={images.shape}")
    print(f"images have dtype={images.dtype}")


def print_label_info(labels):
    """
    @type labels: numpy.ndarray
    """
    print("Type of labels:", type(labels))
    print("First few labels:", labels[:10])
    print("First label (for the first image):", labels[0])


def convert_label_onehot(labels, classes):
    """
    Converts ndarray labels into onehot representation.
    Expects labels as numpy.ndarray and converts them as onehot vec.

    @type labels: numpy.ndarray
    """
    return tf.one_hot(labels, depth=classes)


def save_debug_image(image, label, imagename, greyscale=True):
    """
    Saves an ndarray image into a debug image with given image name.

    @type image: numpy.ndarray item of an image
    @type label: numpy.ndarray item with dtype int
    @type imagename: string
    @type greyscale: boolean
    """
    if greyscale:
        cmap = "gray"
    else:
        cmap = "viridis"
    plt.figure()
    plt.imshow(image, cmap=cmap)
    plt.title(str(label))
    plt.savefig(imagename)


def print_label_balance(labels):
    """
    Prints the label distribution in a single line.
    @type labels: numpy.ndarray
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    label_count_pairs = [
        f"{label}: {count}" for label, count in zip(unique_labels, counts)
    ]
    print("Label distribution:", ", ".join(label_count_pairs))


def preprocess_ndarray(images, labels):
    """preprocesses images and labels and returns a full dataset
    @type images: numpy.ndarray
    @type labels: numpy.ndarray
    """
    images = normalize_convert_images_f32(images)
    labels = convert_label_onehot(labels, classes=10)
    return tf.data.Dataset.from_tensor_slices((images, labels))


def preprocess_dataset(
    dataset,
    batch_size,
    enable_shuffling=True,
    enable_prefetch=True,
    prefetch_buffer=tf.data.AUTOTUNE,
):
    """preprocesses a dataset"""
    if enable_shuffling:
        dataset = full_shuffle_dataset(dataset)
    dataset = batch_dataset(dataset, batch_size)

    # https://www.tensorflow.org/api_docs/python/tf/data/Dataset?_gl=1*37zn2u*_up*MQ..*_ga*NjY5MzU0Mzk0LjE3NDU2NzI2OTE.*_ga_W0YLR4190T*MTc0NTY3MjY5MC4xLjAuMTc0NTY3MjY5MC4wLjAuMA..#prefetch
    # Most dataset input pipelines should end with a call to prefetch.
    # This allows later elements to be prepared while the current element is being processed.
    # This often improves latency and throughput, at the cost of using additional memory to store prefetched elements.
    if enable_prefetch:
        dataset = dataset.prefetch(buffer_size=prefetch_buffer)
    return dataset


def split_dataset(full_dataset, validation_size):
    valid_dataset = full_dataset.take(validation_size)
    train_dataset = full_dataset.skip(validation_size)
    return (train_dataset, valid_dataset)
