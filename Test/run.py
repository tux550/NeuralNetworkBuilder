import numpy as np
from testing.test import test_hyperparameter
from models.files import FILES_CONFIG
from models.mlp import MLP_CONFIG, gen_relu_and_final_model


depths = [1,2,3]
nsize = [50,] #[50,100,200]

for d in depths:
    for n in nsize:        
        nodes_size = [n for _ in range(d)] + [24,]
        # Get model
        mdl = gen_relu_and_final_model(nodes_size, final="sigmoid")
        # TEST ALPHA:
        test_hyperparameter("alpha",[0.1,0.01,0.001], MLP_CONFIG,FILES_CONFIG, f"modeldepth{d}", display_cm=True)


# TEST ALPHA:
#test_hyperparameter("alpha",[0.1,0.01,0.001], MLP_CONFIG,FILES_CONFIG, "model_epochs_{}.mdl", display_cm=True)


