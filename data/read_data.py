import json

# Read data from input file
def read_data(data_distribution_file, data_information_file, num_class=10):
    """
    It will preprocess the data in txt format into the dictionary we want.
    :param source of txt file for device information (EMD (earth mover distance), data quantity of each class, variance)
    :param source of txt file for device information (computing time, transmission time, data quantity)
    :return dict of device information
    """
    with open(data_distribution_file, 'r') as rf:
        device_information = rf.readlines()
        
    with open(data_information_file, 'r') as rf:
        device_information_detail = rf.readlines()

    data_distribution = {}
    each_class = []

    # In cifar10 dataset, there are 10 classes.
    for i in range(num_class):
        each_class.append(0)
        
    for i in range(len(device_information)):
        temp = []
        temp_class = device_information[i].split(';')[1][1:-1].split(' ')
        temp_emd = device_information[i].split(';')[0]
        temp_info = device_information_detail[i][1:-2].split(',')
        temp_var = device_information[i].split(';')[2][:-1]
        
        for j in range(10):
            temp.append([each_class[j], each_class[j] + int(temp_class[j])])
            each_class[j] += int(temp_class[j])
            
        data_distribution[i] = {}
        data_distribution[i]['training time'] = int(temp_info[0])
        data_distribution[i]['transmission time'] = float(temp_info[1])
        data_distribution[i]['data_quantity'] = int(temp_info[2])
        data_distribution[i]['emd'] = float(temp_emd)
        data_distribution[i]['variance'] = float(temp_var)
        data_distribution[i]['data_distribution'] = temp

    return data_distribution

def read_setting():
    """
    Read detailed settings into dictionary
    :return dictionary which contains file source, device list, and training information
    """
    with open('data/detailed_settings.json' , 'r') as reader:
        json_result = json.loads(reader.read())
        
    return json_result