import numpy as np
from utils.metrics import get_load_metrics
from models.files import FILES_CONFIG
from models.mlp import MLP_CONFIG, gen_relu_and_final_model


d = 1
n = 200
alpha = 0.1
final = "sigmoid"

nodes_size = [n for _ in range(d)] + [24,]
# Get model of depth d with n nodes per layer
mdl = gen_relu_and_final_model(nodes_size, final="sigmoid")
mdl["filename_import_model"] = "export/best.mdl"


m = get_load_metrics(mdl |FILES_CONFIG)
print(m)