import cv2
import numpy as np
from sklearn.utils import shuffle
from random import randint
from keras.utils import to_categorical
import random

def random_contrast(im, lower=0.2, upper=1.8):
    """
    It will randomly adjust contrast of input image.
    :param image to be adjusted
    :param contrast lower range (default 0.2)
    :param contrast higher range (default 1.8)
    :return processed image
    """
    prob = randint(0, 1)
    if prob == 1:
        alpha = random.uniform(lower, upper)
        imgg = im * alpha
        imgg = imgg.clip(min=0, max=1)
        return imgg
    else:
        return im

def random_bright(im, delta=63):
    """
    It will randomly adjust brightness of input image.
    :param image to be adjusted
    :param brightness range (default range from -63 to 63)
    :return processed image
    """
    prob = randint(0,1)
    if prob == 1:
        delta = random.uniform(-delta, delta)
        imgg = im + delta / 255.0
        imgg = imgg.clip(min=0, max=1)
        return imgg
    else:
        return im

def per_image_standardization(img):
    """
    It will adjust standardization of input image.
    :param image to be adjusted
    :return processed image
    """
    num_compare = img.shape[0] * img.shape[1] * 3
    img_arr=np.array(img)
    img_t = (img_arr - np.mean(img_arr))/max(np.std(img_arr), 1/num_compare)
    return img_t

def random_crop(img, width, height):
    """
    It will randomly crop input image with width and height.
    :param image to be adjusted
    :param crop width
    :param crop height
    :return processed image
    """
    width1 = randint(0, img.shape[0] - width)
    height1 = randint(0, img.shape[1] - height)
    cropped = img[height1:height1+height, width1:width1+width]

    return cropped

def random_flip_left_right(image):
    """
    It will randomly filp the input image left or right.
    :param image to be flipped
    :return processed image
    """
    prob = randint(0, 1)
    if prob == 1:
        image = np.fliplr(image)
    return image

def preprocessing_for_training(images):
    distorted_image = random_flip_left_right(images)
    distorted_image = random_bright(distorted_image)
    distorted_image = random_contrast(distorted_image)
    float_image = per_image_standardization(distorted_image)
    
    return float_image

def preprocessing_for_testing(images):
    distorted_image = cv2.resize(images, dsize=(24, 24), interpolation=cv2.INTER_CUBIC)
    distorted_image = per_image_standardization(distorted_image)
    
    return distorted_image

# Prepare for the training data with providing the data range (for Gossip Learning)
def prepare_for_training_data_selected_random_range(device_num, data_range, train_images, train_labels):
    image, label = train_images, train_labels

    if data_range[0] < data_range[1]:
        s0 = [image[label[:, 0] == 0][data_range[0]:data_range[1]], label[label[:, 0] == 0][data_range[0]:data_range[1]]]

        for classes in range(1, 10):
            s1 = [image[label[:, 0] == classes][data_range[0]:data_range[1]], label[label[:, 0] == classes][data_range[0]:data_range[1]]]
            s0 = [np.append(s0[0], s1[0], axis=0), np.append(s0[1], s1[1])]

    else:
        s1 = [image[label[:, 0] == 0][data_range[0]:], label[label == 0][data_range[0]:]]
        a = image[label[:, 0] == 0][:data_range[1]]
        a_label = label[label[:, 0] == 0][:data_range[1]]

        s0 = [np.append(a, s1[0], axis=0), np.append(a_label, s1[1])]

        for classes in range(1, 10):
            s1 = [image[label[:, 0] == classes][data_range[0]:], label[label[:, 0] == classes][data_range[0]:]]
            a = image[label[:, 0] == classes][:data_range[1]]
            a_label = label[label == classes][:data_range[1]]
            s1 = [np.append(a, s1[0], axis=0), np.append(a_label, s1[1])]

            s0 = [np.append(s0[0], s1[0], axis=0), np.append(s0[1], s1[1])]

    for _ in range(20):
        s0[0], s0[1] = shuffle(s0[0], s0[1], random_state=randint(0, device_num))
        
    return s0[0], to_categorical(s0[1], 10)

# Prepare for the training data without providing the data range (for Gossip Learning)
def prepare_for_training_data_selected_random(device_num, device_data_amount, train_images, train_labels):

    temp_amount = round(device_data_amount/10)
    image, label = train_images, train_labels
    
    temp_image, temp_label = image[label == 0], label[label == 0]
    shuffle(temp_image, temp_label, random_state=device_num)
    
    s0 = [temp_image[:temp_amount], temp_label[:temp_amount]]
    
    for classes in range(1, 10):
        temp_image, temp_label = image[label == classes], label[label == classes]
        shuffle(temp_image, temp_label, random_state=device_num * classes)

        s1 = [temp_image[:temp_amount], temp_label[:temp_amount]]
        s0 = [np.append(s0[0], s1[0], axis=0), np.append(s0[1], s1[1])]
    
    s0[0], s0[1] = shuffle(s0[0], s0[1], random_state=randint(0, device_num))

    return s0[0], to_categorical(s0[1], 10)

def prepare_for_training_data(device_num, train_images, train_labels, num_device):
    num_data = int(len(train_images)/num_device/10)
    device_num = device_num * num_data
    
    image, label = train_images, train_labels
    
    s0 = [image[label[:, 0] == 0][device_num : device_num+num_data], label[label[:, 0] == 0][device_num : device_num+num_data]]

    for i in range(1, 10):
        s1 = [image[label[:, 0] == i][device_num : device_num+num_data], label[label[:, 0] == i][device_num : device_num+num_data]]

        s0 = [np.concatenate((s0[0], s1[0]), axis=0), np.append(s0[1], s1[1])]

    s0[0], s0[1] = shuffle(s0[0], s0[1], random_state=randint(0, device_num))
    
    return s0[0], to_categorical(s0[1])

def prepare_for_testing_data(device_num, test_images, test_labels, num_device):
    num_data = int(len(test_images)/num_device/10)
    device_num = device_num * num_data

    image, label = test_images, test_labels
#     image, label = shuffle(test_images, test_labels, random_state=0)
    
    s0 = [image[label[:, 0] == 0][device_num : device_num+num_data], label[label[:, 0] == 0][device_num : device_num+num_data]]

    for i in range(1, 10):
        s1 = [image[label[:, 0] == i][device_num : device_num+num_data], label[label[:, 0] == i][device_num : device_num+num_data]]

        s0 = [np.concatenate((s0[0], s1[0]), axis=0), np.append(s0[1], s1[1])]

    s0[0], s0[1] = shuffle(s0[0], s0[1], random_state=randint(0, device_num))
    
    return s0[0], to_categorical(s0[1])

def prepare_for_training_data_mnist(device_num, train_images, train_labels, num_device):
    num_data = int(len(train_images)/num_device/10)
    device_num = device_num * num_data

    image, label = train_images, train_labels
    
    s0 = [image[label == 0][device_num : device_num + num_data], label[label == 0][device_num : device_num + num_data]]

    for i in range(1, 10):
        s1 = [image[label == i][device_num : device_num+num_data], label[label == i][device_num : device_num+num_data]]

        s0 = [np.concatenate((s0[0], s1[0]), axis=0), np.append(s0[1], s1[1])]

    s0[0], s0[1] = shuffle(s0[0], s0[1], random_state=randint(0, device_num))
    
    return s0[0], to_categorical(s0[1], 10)


def prepare_for_testing_data_mnist(device_num, test_images, test_labels, num_device):
    num_data = int(len(test_images)/num_device/10)
    device_num = device_num * num_data

    image, label = test_images, test_labels
#     image, label = shuffle(test_images, test_labels, random_state=0)
    
    s0 = [image[label == 0][device_num : device_num+num_data], label[label == 0][device_num : device_num+num_data]]

    for i in range(1, 10):
        s1 = [image[label == i][device_num : device_num+num_data], label[label == i][device_num : device_num+num_data]]

        s0 = [np.concatenate((s0[0], s1[0]), axis=0), np.append(s0[1], s1[1])]

    s0[0], s0[1] = shuffle(s0[0], s0[1], random_state=randint(0, device_num))
    
    return s0[0], to_categorical(s0[1], 10)
