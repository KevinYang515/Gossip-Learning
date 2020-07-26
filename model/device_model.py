import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

class Device_Model():
    def __init__(self, number):
        self.number = number
        self.weight = define_model()
        self.weight.compile(optimizer='sgd',
                                  loss='categorical_crossentropy',
                                  metrics=['accuracy'])
        self.weight.load_weights('model/weight/ini_cifar10.h5')
        
        self.client = {}
        self.history = {}
        self.first = {}
        self.master_first = True
        self.weight_hat = define_model()
        self.weight_q = define_model()
        
        self.weight_list = []
        
    def add_client(self, client, degree):
        if client not in self.client:
            self.client[client] = {}
            self.client[client]['weight'] = define_model()
            self.client[client]['degree'] = degree
            self.first[client] = True
        else:
            print(client, "has already existed in client dictionary.")
        
    def add_client_list(self, client, client_dictionary):
        for cli in client:
            if cli not in self.client:
                self.client[cli] = {}
                self.client[cli]['weight'] = define_model()
                self.client[cli]['degree'] = len(client_dictionary[cli])
                self.first[cli] = True
            else:
                print(cli, "has already existed in client dictionary.")
        
    def set_client_weight(self, client, value):
        if client in self.client:
            if self.first[client] == True:
                self.first[client] = False
                self.client[client]['weight'].set_weights(value.get_weights())
            else:
                for layer in range(len(self.weight.layers)):
                    self.client[client]['weight'].layers[layer].set_weights(
                        np.add(value.layers[layer].get_weights(),
                          self.client[client]['weight'].layers[layer].get_weights()))
        else:
            print(client, "doesn't exist in client dictionary.")
    
    def update_own_q(self):
        if self.master_first == True:
            self.master_first = False
            self.weight_hat.set_weights(self.weight_q.get_weights())
        else:
            for layer in range(len(self.weight.layers)):
                self.weight_hat.layers[layer].set_weights(
                    np.add(self.weight_q.layers[layer].get_weights(), 
                      self.weight_hat.layers[layer].get_weights()))
            
    def count_q(self):
        if self.master_first == True:
            self.weight_q.set_weights(self.weight.get_weights())
        else:
            for layer in range(len(self.weight.layers)):
                self.weight_q.layers[layer].set_weights(
                    np.subtract(self.weight.layers[layer].get_weights(), 
                    self.weight_hat.layers[layer].get_weights()))
    
    def update_parameter(self):
        if self.master_first == False:
            client_list = [x for x in self.client]
            
            temp = {}
            
            for device in client_list:
                client_weight = self.weight_list[device] * 0.5
                
                if client_list.index(device) == 0:
                    for layer in range(len(self.weight.layers)):
                        temp[layer] = np.subtract(self.client[device]['weight'].layers[layer].get_weights(), 
                                    self.weight_hat.layers[layer].get_weights())
                elif client_list.index(device) == len(client_list) - 1:
                    for layer in range(len(self.weight.layers)):
                        self.weight.layers[layer].set_weights(
                            np.add(self.weight.layers[layer].get_weights(),
                              np.multiply(np.add(temp[layer], 
                                  np.subtract(self.client[device]['weight'].layers[layer].get_weights(), 
                                    self.weight_hat.layers[layer].get_weights())), client_weight)))
                                
                else:
                    for layer in range(len(self.weight.layers)):
                        temp[layer] = np.add(temp[layer], 
                                        np.subtract(self.client[device]['weight'].layers[layer].get_weights(), 
                                          self.weight_hat.layers[layer].get_weights()))
                        
    def set_weight(self, device_dictionary, num_node):
        self.weight_list = generate_weight(self.number, device_dictionary, num_node)

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

# Define our model
def define_model():
    """
    Define the architecture of our model from the paper of Google
    :return model architecture 
    """
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
