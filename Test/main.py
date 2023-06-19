import numpy as np
from testing.test import test_hyperparameter
from config_iris import *
from config_ht import *

default_arch = {
    # ARCHITECTURE
    "input_size": 4,
    "nodes_size": [40,40,3],
    "activation_function":["hypertan","hypertan","hypertan"],
    "loss_function":"mse",
}
default_train = {
    # TRAIN
    "filename_x_train":"dataset/x_train.csv",
    "filename_y_train":"dataset/y_train.csv",
    "epochs":5_000,
    "alpha":0.01,
}

default_test = {
    # TEST
    "filename_x_test":"dataset/x_test.csv",
    "filename_y_test":"dataset/y_test.csv",
}


#test_hyperparameter("nodes",[10,20,40,60,100],default_arch_ht, default_train_ht, default_test_ht, "model_bs_{}.mdl", display_cm=True)
test_hyperparameter("alpha",[0.01,],default_arch_ht, default_train_ht, default_test_ht, "model_epochs_{}.mdl", display_cm=True)
#test_hyperparameter("epochs",[50,],iris_arch, iris_train, iris_test, "model_iris_{}.mdl", display_cm=True)

