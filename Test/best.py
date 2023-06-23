import numpy as np
from utils.metrics import get_metrics
from models.files import FILES_CONFIG
from models.mlp import MLP_CONFIG, gen_relu_and_final_model


d = 1
n = 200
alpha = 0.1
final = "sigmoid"

nodes_size = [n for _ in range(d)] + [24,]
# Get model of depth d with n nodes per layer
mdl = gen_relu_and_final_model(nodes_size, final="sigmoid")
mdl["epochs"] = 1000
mdl["alpha"] = alpha


m = get_metrics(mdl |FILES_CONFIG, "best")
print(m)