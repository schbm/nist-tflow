from data import DataLoaderMNIST
from model import MyModel
import tensorflow as tf
import keras
import time
import datetime
import numpy as np


class Trainer:
    def __init__(self, model):
        self.model = model
        self.optimizer = keras.optimizers.Adam(learning_rate=0.001)
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy()

        self.train_loss_metric = tf.keras.metrics.Mean("train_loss", dtype=tf.float32)
        self.train_acc_metric = keras.metrics.CategoricalAccuracy()

        self.valid_loss_metric = tf.keras.metrics.Mean("valid_loss", dtype=tf.float32)
        self.val_acc_metric = keras.metrics.CategoricalAccuracy()

        self.max_saved_misclassified_images = 20
        self.misclassified_images = []

    @tf.function
    def train_step(self, x_batch_train, y_batch_train):
        with tf.GradientTape() as tape:
            pred = self.model(x_batch_train, training=True)
            loss_value = self.loss_fn(y_batch_train, pred)

        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        self.train_loss_metric(loss_value)
        self.train_acc_metric.update_state(y_batch_train, pred)

    def val_step(self, x_batch_val, y_batch_val, save_missclassified=True):
        pred = self.model(x_batch_val, training=False)
        loss = self.loss_fn(y_batch_val, pred)

        if (
            save_missclassified
            and len(self.misclassified_images) < self.max_saved_misclassified_images
        ):
            self.save_missclassified(x_batch_val, y_batch_val, pred)

        self.valid_loss_metric(loss)
        self.val_acc_metric.update_state(y_batch_val, pred)

    def save_missclassified(self, x_batch_val, y_batch_val, pred):
        pred_classes_np = np.argmax(pred, axis=-1)

        labels_np = np.argmax(y_batch_val, axis=1)

        images_np = x_batch_val.numpy()

        misclassified_indices = np.where(pred_classes_np != labels_np)[0]

        misclassified_images = images_np[misclassified_indices]
        misclassified_true_labels = labels_np[misclassified_indices]
        misclassified_pred_labels = pred_classes_np[misclassified_indices]

        for i in range(
            min(
                self.max_saved_misclassified_images - len(self.misclassified_images),
                len(misclassified_images),
            )
        ):
            self.misclassified_images.append(
                (
                    misclassified_images[i],  # Misclassified image
                    misclassified_true_labels[i],  # True label
                    misclassified_pred_labels[i],  # Predicted label
                )
            )

    def __call__(self, train_dataset, valid_dataset, epochs):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = "logs/gradient_tape/" + current_time + "/train"
        valid_log_dir = "logs/gradient_tape/" + current_time + "/valid"

        summary_writer = tf.summary.create_file_writer(train_log_dir)
        valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)

        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))
            start_time = time.time()

            # Training loop
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                if epoch == 0 and step == 0:
                    tf.summary.trace_on(graph=True, profiler=True)
                self.train_step(x_batch_train, y_batch_train)
                if epoch == 0 and step == 0:
                    print("Exporting model graph...")
                    with summary_writer.as_default():
                        tf.summary.trace_export(
                            name="model_graph",
                            step=0,
                            profiler_outdir=train_log_dir,
                        )

            with summary_writer.as_default():
                tf.summary.scalar("loss", self.train_loss_metric.result(), step=epoch)
                tf.summary.scalar(
                    "accuracy", self.train_acc_metric.result(), step=epoch
                )
                for weight in self.model.trainable_weights:
                    tf.summary.histogram(weight.name, weight, step=epoch)

            # Validation loop
            for x_batch_val, y_batch_val in valid_dataset:
                self.val_step(x_batch_val, y_batch_val, save_missclassified=False)
            with valid_summary_writer.as_default():
                tf.summary.scalar("loss", self.valid_loss_metric.result(), step=epoch)
                tf.summary.scalar("accuracy", self.val_acc_metric.result(), step=epoch)

            template = (
                "Epoch {}, Loss: {}, Accuracy: {}, Valid Loss: {}, Valid Accuracy: {}"
            )
            print(
                template.format(
                    epoch + 1,
                    self.train_loss_metric.result(),
                    self.train_acc_metric.result() * 100,
                    self.valid_loss_metric.result(),
                    self.val_acc_metric.result() * 100,
                )
            )
            print("Time taken: %.2fs" % (time.time() - start_time))

            self.train_loss_metric.reset_state()
            self.train_acc_metric.reset_states()
            self.valid_loss_metric.reset_state()
            self.val_acc_metric.reset_states()

        # final evaluation with missclassified images
        for x_batch_val, y_batch_val in valid_dataset:
            self.val_step(x_batch_val, y_batch_val, save_missclassified=True)
        self.log_misclassified_images(valid_summary_writer)

    def log_misclassified_images(self, summary_writer):
        with summary_writer.as_default():
            for i, (image, true_label, pred_label) in enumerate(
                self.misclassified_images
            ):

                print("Shape: ", image.shape)
                print("Label: ", true_label, "->", pred_label)
                reshaped_image = np.expand_dims(
                    image, axis=0
                )  # Adds the batch dimension
                reshaped_image = np.expand_dims(
                    reshaped_image, axis=-1
                )  # Adds the channel dimension
                tf.summary.image(
                    f"Training data {pred_label} -> {true_label}",
                    reshaped_image,
                    step=0,
                )


if __name__ == "__main__":
    model = MyModel("MNISTClassifier")

    data_loader = DataLoaderMNIST()
    train_dataset = data_loader.train_dataset
    valid_dataset = data_loader.valid_dataset

    train = Trainer(model)
    train(train_dataset, valid_dataset, epochs=40)

    model.save("my_model.keras")
