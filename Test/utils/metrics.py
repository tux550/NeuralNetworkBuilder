import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix
from .run import run_train_test_save
from .np import numpy_from_file

def metrics(arch_dict, train_dict, test_dict, save_path):
    # Create parameters dictionary
    parameters_dict = arch_dict | train_dict | test_dict
    parameters_dict["filename_export_model"]=save_path
    # Y PRED
    y_pred = run_train_test_save(parameters_dict)
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