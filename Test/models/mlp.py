MLP_CONFIG = {
    # ARCHITECTURE
    "input_size": 128,
    "nodes_size": [60,60,60,60,60,24],
    "activation_function":["relu","relu","relu","relu","relu","sigmoid"],
    "loss_function":"cross_entropy",
    # TRAIN
    "epochs":100,
    "alpha":0.001,
}

