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

def read_simple_setting():
    """
    Read detailed settings into dictionary
    :return dictionary which contains file source, device list, and training information
    """
    with open('data/simple_detailed_settings.json' , 'r') as reader:
        json_result = json.loads(reader.read())
        
    return json_result

def read_setting():
    """
    Read detailed settings into dictionary
    :return dictionary which contains file source, device list, and training information
    """
    with open('data/detailed_settings.json' , 'r') as reader:
        json_result = json.loads(reader.read())
        
    return json_result

def read_data_gossip(file_name):
    with open('data/file/graph_small/small_graph_0501/' + file_name, 'r') as f:
        lines = f.readlines()
        
    device_client_dic = {}
    for i in [y.split('\t')[:2] for y in [x for x in lines[1:]]]:
        first_temp = int(i[0])
        second_temp= int(i[1])

        if first_temp not in device_client_dic:
            device_client_dic[first_temp] = []
        if second_temp not in device_client_dic:
            device_client_dic[second_temp] = []

        device_client_dic[first_temp].append(second_temp)
        device_client_dic[second_temp].append(first_temp)
    sort_dictionary(device_client_dic)
    return device_client_dic

def sort_dictionary(dic):
    temp = sorted(dic.keys())
    temp_dic = {}
    
    for key in temp:
        temp_dic[key] = dic[key]
    
    return temp_dic