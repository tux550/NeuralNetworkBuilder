
import subprocess
import numpy as np
from sklearn.metrics import confusion_matrix

def numpy_from_file(filename):
    return np.genfromtxt(filename, delimiter=' ')

def numpy_from_str(txt):
    return np.array([[float(e) for e in row.strip(" ").split(" ")] for row in txt.split("\n")])

def run_exe(parameters_dict):
    # CONSTANTS
    KEY_ORDER = [
        "input_size",
        "output_size",
        "depth",
        "nodes",
        "activation_function",
        "loss_function",
        "filename_x",
        "filename_y",
        "epochs",
        "alpha",
        "batch_size"
    ]
    EXECUTABLE = "../NeuralNetwork/mlp.exe"
    CMD = 'echo "{parameters}" | {exe_file}'
    # FORMAT
    parameters = " ".join([ str(parameters_dict[k]) for k in KEY_ORDER])
    cmd = CMD.format(parameters=parameters, exe_file=EXECUTABLE)
    # RUN
    print("Running with parameters:", parameters)
    y_pred =subprocess.getoutput(cmd)
    y_pred =numpy_from_str(y_pred)
    return y_pred


def metrics(parameters_dict):
    # Y PRED
    y_pred = run_exe(parameters_dict)
    y_pred = np.argmax(y_pred, axis=1)
    # Y TRUE
    y_true = numpy_from_file(parameters_dict["filename_y"])
    y_true = np.argmax(y_true, axis=1)
    # Generate Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("CONFUSION MATRIX")
    print(cm)


















default = {
    "input_size" : 4,
    "output_size" : 3,
    "depth" : 2,
    "nodes" : 40,
    "activation_function" : "hypertan",
    "loss_function" : "mse",
    "filename_x" : "../Dataset/x.csv",
    "filename_y" : "../Dataset/y.csv",
    "epochs" :10000,
    "alpha" : 0.01,
    "batch_size" : 4,
}


metrics(default)
