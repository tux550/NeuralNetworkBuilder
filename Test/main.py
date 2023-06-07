import numpy as np
from testing.test import test_hyperparameter


default_arch = {
    # ARCHITECTURE
    "input_size":32,
    "output_size":24,
    "depth":6,
    "nodes":100,
    "activation_function":"relu",
    "loss_function":"mse",
}
default_train = {
    # TRAIN
    "filename_x_train":"dataset/x_train.csv",
    "filename_y_train":"dataset/y_train.csv",
    "epochs":100_000, #5_000,
    "alpha":0.00001,
    "batch_size":1,
}

default_test = {
    # TEST
    "filename_x_test":"dataset/x_train.csv",
    "filename_y_test":"dataset/y_train.csv",
}


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


test_hyperparameter("batch_size",[1,],default_arch, default_train, default_test, "model_bs_{}.mdl", display_cm=True)
#test_hyperparameter("batch_size",[1,],iris_arch, iris_train, iris_test, "model_iris_{}.mdl", display_cm=True)

