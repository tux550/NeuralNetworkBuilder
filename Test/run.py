import numpy as np
from testing.test import test_hyperparameter
from models.files import FILES_CONFIG
from models.mlp import MLP_CONFIG


# TEST ALPHA:
test_hyperparameter("alpha",[0.1,0.01,0.001], MLP_CONFIG,FILES_CONFIG, "model_epochs_{}.mdl", display_cm=True)


