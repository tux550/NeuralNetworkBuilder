from utils.metrics import get_metrics
from .table import table_results

def test_hyperparameter(hp_name, hp_list, mlp_dict, files_dict, name_template, display_cm=False):
    # Results
    results = dict()
    # For each value
    for hp in hp_list:
        # Set Hyperparemeter
        if hp_name in mlp_dict:
            mlp_dict[hp_name] = hp
        else:
            raise f"Hyperparameter {hp_name} does not exist"
        # Get metrics
        metrics = get_metrics(mlp_dict | files_dict, name_template.format(str(hp)), display_cm=display_cm)
        # Save to results
        results[hp] = metrics
    # Display
    table_results(hp_name, results)

