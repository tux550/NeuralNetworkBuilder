import os
import subprocess
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix

def numpy_from_file(filename):
    return np.genfromtxt(filename, delimiter=' ')

def numpy_from_str(txt):
    return np.array([[float(e) for e in row.strip(" ").split(" ")] for row in txt.split("\n")])

def run_exe(parameters_dict):
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


def metrics(parameters_dict):
    # Y PRED
    y_pred = run_exe(parameters_dict)
    y_pred = np.argmax(y_pred, axis=1)
    # Y TRUE
    y_true = numpy_from_file(parameters_dict["filename_y_test"])
    y_true = np.argmax(y_true, axis=1)
    # Generate Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("CONFUSION MATRIX")
    print(cm)

    metrics = {
        "Balanced Accuracy" : balanced_accuracy_score(y_true, y_pred),
        "Precision" : precision_score(y_true, y_pred, average=None),
        "Recall" : recall_score(y_true, y_pred, average=None),
        "F1 Score" : f1_score(y_true, y_pred, average=None),
    }

    print(metrics)

default = {
    # MODE
    "train_load":1,
    "test":1,
    "save":1,
    # ARCHITECTURE
    "input_size":4,
    "output_size":3,
    "depth":2,
    "nodes":40,
    "activation_function":"hypertan",
    "loss_function":"mse",
    # TRAIN
    "filename_x_train":"../Dataset/x.csv",
    "filename_y_train":"../Dataset/y.csv",
    "epochs":10000,
    "alpha":0.01,
    "batch_size":4,
    # TEST
    "filename_x_test":"../Dataset/x.csv",
    "filename_y_test":"../Dataset/y.csv",
    # SAVE
    "filename_export_model":"model.mdl",
}


metrics(default)
