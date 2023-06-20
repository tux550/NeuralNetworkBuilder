import numpy as np
from testing.test import test_hyperparameter
from models.files import FILES_CONFIG
from models.mlp import MLP_CONFIG, gen_relu_and_final_model


depths = [1,2,3]
nsize = [50,100,200]

# SKIP
#depth 1 completed
#depth 2 missing hypertan 100 + both 200
#depth 3 all missing

alpha_ls = [0.1,0.01,0.001]

for d in depths:
    for n in nsize:

        # SKIP COMPLETED
        if (d==1) or (d==2 and n==50):
            continue
        # ----

        nodes_size = [n for _ in range(d)] + [24,]
        # Get model of depth d with n nodes per layer
        mdl = gen_relu_and_final_model(nodes_size, final="hypertan")
        # TEST ALPHA:
        test_hyperparameter("alpha",alpha_ls, MLP_CONFIG,FILES_CONFIG, f"reluhypertan_modeldepth{d}_nsize{n}", display_cm=True)

        # SKIP COMPLETED
        if (d==2 and n==100):
            continue
        # ----

        # Get model of depth d with n nodes per layer
        mdl = gen_relu_and_final_model(nodes_size, final="sigmoid")
        # TEST ALPHA:
        test_hyperparameter("alpha",alpha_ls, MLP_CONFIG,FILES_CONFIG, f"relusigmoid_modeldepth{d}_nsize{n}", display_cm=True)



