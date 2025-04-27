import tensorflow as tf
import keras
from data import DataLoaderMNIST
from model import MyModel
import datetime
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class Tester:
    def __init__(self, model):
        self.model = model

    def __call__(self, test_dataset):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        test_log_dir = "logs/gradient_tape/" + current_time + "/test"
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        test_loss, test_acc = self.model.evaluate(test_dataset)

        with test_summary_writer.as_default():
            tf.summary.scalar("test_loss", test_loss, step=0)
            tf.summary.scalar("test_accuracy", test_acc, step=0)

        print("Test Loss:", test_loss)
        print("Test Accuracy:", test_acc)

        true_labels = []
        predictions = []

        for images, labels in test_dataset:
            preds = self.model.predict(images)
            true_labels.extend(np.argmax(labels, axis=1))
            predictions.extend(np.argmax(preds, axis=1))

        cm = confusion_matrix(true_labels, predictions)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(10), yticklabels=np.arange(10))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()


if __name__ == "__main__":
    model = keras.models.load_model("my_model.keras")

    data_loader = DataLoaderMNIST()
    test_dataset = data_loader.test_dataset

    test = Tester(model)
    test(test_dataset)
