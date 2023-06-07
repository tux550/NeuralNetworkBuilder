import numpy as np
from testing.test import test_hyperparameter


default_arch = {
    # ARCHITECTURE
    "input_size":4,
    "output_size":3,
    "depth":2,
    "nodes":40,
    "activation_function":"hypertan",
    "loss_function":"mse",
}
default_train = {
    # TRAIN
    "filename_x_train":"../Dataset/x.csv",
    "filename_y_train":"../Dataset/y.csv",
    "epochs":10000,
    "alpha":0.01,
    "batch_size":4,
}

default_test = {
    # TEST
    "filename_x_test":"../Dataset/x.csv",
    "filename_y_test":"../Dataset/y.csv",
}

test_hyperparameter("alpha",[0.1,0.01,0.001],default_arch, default_train, default_test, "model_hp_{}.mdl")
#get_metrics(default_arch, default_train, default_test, "model_test.mdl")
