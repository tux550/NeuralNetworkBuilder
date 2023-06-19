
default_arch_ht = {
    # ARCHITECTURE
    "input_size": 128,
    "nodes_size": [60,60,60,60,60,24],
    "activation_function":["relu","relu","relu","relu","relu","sigmoid"],
    #"activation_function":["relu","relu","relu","relu","relu","hypertan"],
    #"nodes_size": [100,100,24],
    #"activation_function":["hypertan","hypertan","hypertan"],
    "loss_function":"cross_entropy",
    #"loss_function":"mse",
}
default_train_ht = {
    # TRAIN
    "filename_x_train":"dataset/x_train.csv",
    "filename_y_train":"dataset/y_train.csv",
    "epochs":100,
    "alpha":0.001,
}

default_test_ht = {
    # TEST
    "filename_x_test":"dataset/x_test.csv",
    "filename_y_test":"dataset/y_test.csv",
}
