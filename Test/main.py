import numpy as np
from testing.test import test_hyperparameter
from config_iris import *
from config_ht import *

default_arch = {
    # ARCHITECTURE
    "input_size":32,
    "output_size":24,
    "depth":2,#"depth":1,
    "nodes":20,
    "activation_function":"hypertan",
    "loss_function":"mse",
}
default_train = {
    # TRAIN
    "filename_x_train":"dataset/x_train.csv",
    "filename_y_train":"dataset/y_train.csv",
    "epochs":100_000,
    "alpha":0.01,
    "batch_size":1,
}

default_test = {
    # TEST
    "filename_x_test":"dataset/x_test.csv",
    "filename_y_test":"dataset/y_test.csv",
}


test_hyperparameter("nodes",[10,20,40,60,100],default_arch_ht, default_train_ht, default_test_ht, "model_bs_{}.mdl", display_cm=True)
#test_hyperparameter("batch_size",[1,],iris_arch, iris_train, iris_test, "model_iris_{}.mdl", display_cm=True)

