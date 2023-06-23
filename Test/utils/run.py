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
    "depth",
    "network_arch",
    "loss_function",
    # TRAIN
    "filename_x_train",
    "filename_y_train",
    "epochs",
    "alpha",
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
    # SET Network arch
    parameters_dict["depth"] = len(parameters_dict["nodes_size"])+1
    parameters_dict["network_arch"] = str(parameters_dict["input_size"]) + " "  + " ".join(
        [
            str(n)+" "+str(f) for n,f in zip(
                parameters_dict["nodes_size"],
                parameters_dict["activation_function"]
            )
        ])
    

    # FORMAT
    parameters = " ".join([ str(parameters_dict[k]) for k in KEY_ORDER])
    cmd = CMD.format(parameters=parameters, exe_file=EXECUTABLE)
    # RUN
    print("Running with parameters:", parameters)
    print("cmd:", cmd)
    if os.name == "nt":
        # Windows
        y_pred =subprocess.getoutput(['powershell',cmd])
    else:
        # Linux
        y_pred =subprocess.getoutput(cmd)
    try:
        y_pred =numpy_from_str(y_pred)
    except Exception as e:
        print("EXCEPTION")
        print(y_pred)
        raise e
    return y_pred

# CONSTANTS
KEY_ORDER_LOAD = [
    # MODE
    "train_load",
    "test",
    "save",
    # ARCHITECTURE
    "depth",
    "network_arch",
    "loss_function",
    # LOAD
    "filename_import_model",
    # TEST
    "filename_x_test",
]
def run_load_test(parameters_dict):
    # SET OPTIONS
    parameters_dict["train_load"]=0
    parameters_dict["test"]=1
    parameters_dict["save"]=0
    # SET Network arch
    parameters_dict["depth"] = len(parameters_dict["nodes_size"])+1
    parameters_dict["network_arch"] = str(parameters_dict["input_size"]) + " "  + " ".join(
        [
            str(n)+" "+str(f) for n,f in zip(
                parameters_dict["nodes_size"],
                parameters_dict["activation_function"]
            )
        ])
    # FORMAT
    parameters = " ".join([ str(parameters_dict[k]) for k in KEY_ORDER_LOAD])
    cmd = CMD.format(parameters=parameters, exe_file=EXECUTABLE)
    # RUN
    print("Running with parameters:", parameters)
    print("cmd:", cmd)
    if os.name == "nt":
        # Windows
        y_pred =subprocess.getoutput(['powershell',cmd])
    else:
        # Linux
        y_pred =subprocess.getoutput(cmd)
    try:
        y_pred =numpy_from_str(y_pred)
    except Exception as e:
        print("EXCEPTION")
        print(y_pred)
        raise e
    return y_pred