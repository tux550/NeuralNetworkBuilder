import os
import subprocess
from .np import numpy_from_str

# CONSTANTS
KEY_ORDER = [
    # MODE
    "train_load",
    "test",
    "save",
    # ARCHITECTURE
    "input_size",
    "output_size",
    "depth",
    "nodes",
    "activation_function",
    "loss_function",
    # TRAIN
    "filename_x_train",
    "filename_y_train",
    "epochs",
    "alpha",
    "batch_size",
    # TEST
    "filename_x_test",
    # SAVE
    "filename_export_model"
]
EXECUTABLE = "../NeuralNetwork/mlp.exe"
CMD = 'echo "{parameters}" | {exe_file}'


def run_train_test_save(parameters_dict):
    # SET OPTIONS
    parameters_dict["train_load"]=1
    parameters_dict["test"]=1
    parameters_dict["save"]=1
    # FORMAT
    parameters = " ".join([ str(parameters_dict[k]) for k in KEY_ORDER])
    cmd = CMD.format(parameters=parameters, exe_file=EXECUTABLE)
    # RUN
    print("Running with parameters:", parameters)
    if os.name == "nt":
        # Windows
        y_pred =subprocess.getoutput(['powershell',cmd])
    else:
        # Linux
        y_pred =subprocess.getoutput(cmd)
    y_pred =numpy_from_str(y_pred)
    return y_pred