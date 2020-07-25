from tensorflow.keras import datasets
from keras.utils import to_categorical

# Load CIFAR-10 data
def load_cifar10_data():
    """
    It will load cifar10 dataset.
    :return train_images, train_label, test_images, test_labels
    """
    # If we haven't downloaded cifar10 dataset, it will automatically download it for us.
    # Load CIFAR-10 data
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    return train_images, train_labels, test_images, test_labels

# Transfer train and test label to be categorical
def train_test_label_to_categorical(train_labels, test_labels):
    """
    It will preprocess the data in txt format into the dictionary we want.
    :param labels for training without category
    :param labels for testing without category
    :return categorical labels of training and testing
    """
    # Transfer to be categorical with tensorflow.keras
    train_label = to_categorical(train_labels)
    test_label = to_categorical(test_labels)
    return train_label, test_label