iris_arch = {
    # ARCHITECTURE
    "input_size": 4,
    "nodes_size": [60,60,60,3],
    "activation_function":["hypertan","hypertan","hypertan","hypertan"],
    "loss_function":"mse",
}
iris_train = {
    # TRAIN
    "filename_x_train":"../Dataset/x.csv",
    "filename_y_train":"../Dataset/y.csv",
    "epochs":50,
    "alpha":0.001,
    "batch_size":1,
}

iris_test = {
    # TEST
    "filename_x_test":"../Dataset/x.csv",
    "filename_y_test":"../Dataset/y.csv",
}