def generate_ring(num_node):
    """
    It will generate the ring relationship.
    :param number of nodes
    :return dictionary for relationship of each node
    """
    device_client_dic = {}

    for n in range(num_node):
        next_node = (n + 1) % num_node
        last_node = n - 1
        
        if last_node == -1:
            last_node = num_node - 1
            device_client_dic[n] = [next_node, last_node]
            continue
            
        device_client_dic[n] = [last_node, next_node]

    return device_client_dic