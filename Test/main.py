
import subprocess

def run_test(parameters_dict):
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

    parameters = " ".join([ str(parameters_dict[k]) for k in KEY_ORDER])
    cmd = CMD.format(parameters=parameters, exe_file=EXECUTABLE)

    print("Running with parameters:", parameters)
    result=subprocess.getoutput(cmd)
    return result





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


res = run_test(default)
print("RESULT:")
print(res)