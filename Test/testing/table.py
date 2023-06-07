import numpy as np
from rich.console import Console
from rich.table import Table

def table_results(parameter_name, results_dict):
    title = f"Test Parameter: {parameter_name}"
    table = Table(title=title)
    table.add_column(f"{parameter_name}", justify="right", style="cyan", no_wrap=True)

    # Get parameter and  metrics ls
    p_ls      = results_dict.keys()
    metric_ls = results_dict[next(iter(results_dict))].keys()

    # Create column for each metric
    for metric in metric_ls:
        table.add_column(f"{metric}", justify="center", style="green")

    # Populate table
    for p in p_ls:
        row_p = [str(p),]
        for metric in metric_ls:
            # Get results
            res = results_dict[p][metric]
            # To string
            if type(res) in (np.float32, np.float64, float):
                val = f"{res:.5f}"
            else:
                val = str(res)
            # Add to row
            row_p.append(val)
        table.add_row(*row_p)  

    console = Console(record=True)
    console.print(table)
    console.save_svg(f"fig/table_{parameter_name}.svg")