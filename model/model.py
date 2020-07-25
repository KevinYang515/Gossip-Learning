import tensorflow as tf
from tensorflow.keras import layers, models

def generate_weight(node, device_client_dic, num_node):
    temp = []
    
    for j in range(num_node):
        if j in device_client_dic[node]:
            temp.append(1/(max(len(device_client_dic[node]), len(device_client_dic[j])) + 1))
        else:
            temp.append(0)
            
    temp_v = 0    
    for j in device_client_dic[node]:
        temp_v += temp[j]
        
    temp[node] = 1 - temp_v
    
    return temp

def define_model():
    return tf.keras.Sequential([
        layers.Conv2D(64, (5, 5), padding='same', activation='relu', input_shape=(24, 24, 3)),
        layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (5,5), padding='same', activation='relu'),
        layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(384, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(192, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])