# Gossip-Learning

This is a simple reproduction to partly implement and simulate the paper of [Gossip Learning as a Decentralized Alternative to Federated Learning](https://link.springer.com/chapter/10.1007/978-3-030-22496-7_5 "Gossip Learning").

![image](https://github.com/KevinYang515/Gossip-Learning/blob/master/picture/gossip_learning_fig.PNG)

## Example for Simple Gossip Learning

    python simple_gos_learning.py

We can set how many devices we want.

    python simple_gos_learning.py 10

If you want to adjust more detailed settings, you can modify the json file (i.e., data/detailed_settings.json)

## Example for Gossip Learning Plus

    python gos_learning_plus.py

If you want to change the relationship, you can modify the text file (i.e., data/file/graph_smalle/small_graph_0501). Futhermore, if you want to adjust more detailed settings, you can modify the json file (i.e., data/detailed_settings.json)