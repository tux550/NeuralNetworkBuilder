iris_arch = {
    # ARCHITECTURE
    "input_size":4,
    "output_size":3,
    "depth":4,
    "nodes":40,
    "activation_function":"relu",
    "loss_function":"mse",
}
iris_train = {
    # TRAIN
    "filename_x_train":"../Dataset/x.csv",
    "filename_y_train":"../Dataset/y.csv",
    "epochs":50_000,
    "alpha":0.001,
    "batch_size":1,
}

iris_test = {
    # TEST
    "filename_x_test":"../Dataset/x.csv",
    "filename_y_test":"../Dataset/y.csv",
}