MLP_CONFIG = {
    # ARCHITECTURE
    "input_size": 128,
    "nodes_size": [60,60,60,60,60,24],
    "activation_function":["relu","relu","relu","relu","relu","sigmoid"],
    "loss_function":"cross_entropy",
    # TRAIN
    "epochs":1000,
    "alpha":0.01,
}

def gen_relu_and_final_model(nodes_size, final="sigmoid", default_config=MLP_CONFIG):
    # Update arch
    default_config["nodes_size"] = nodes_size
    default_config["activation_function"] = ["relu" for i in range(len(nodes_size)-1)] + [final,]
    # Return
    return default_config


