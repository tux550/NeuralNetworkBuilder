from utils.metrics import get_metrics
from .table import table_results

def test_hyperparameter(hp_name, hp_list,arch_config, train_config, test_config, name_template, display_cm=False):
    # Results
    results = dict()
    # For each value
    for hp in hp_list:
        # Set Hyperparemeter
        if hp_name in arch_config:
            arch_config[hp_name] = hp
        elif hp_name in train_config:
            train_config[hp_name] = hp
        else:
            raise "Hyperparameter {hp_name} does not exist"
        # Get metrics
        metrics = get_metrics(arch_config, train_config, test_config, name_template.format(str(hp)), display_cm=display_cm)
        # Save to results
        results[hp] = metrics
    # Display
    table_results(hp_name, results)

