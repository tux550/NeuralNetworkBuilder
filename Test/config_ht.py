
default_arch_ht = {
    # ARCHITECTURE
    "input_size":32,
    "output_size":24,
    "depth":2,
    "nodes":20,
    "activation_function":"hypertan",
    "loss_function":"mse",
}
default_train_ht = {
    # TRAIN
    "filename_x_train":"dataset/x_train.csv",
    "filename_y_train":"dataset/y_train.csv",
    "epochs":100_000,
    "alpha":0.01,
    "batch_size":1,
}

default_test_ht = {
    # TEST
    "filename_x_test":"dataset/x_test.csv",
    "filename_y_test":"dataset/y_test.csv",
}
