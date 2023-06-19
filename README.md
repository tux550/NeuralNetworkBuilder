# IA-p3

## Notes
- Hypertan tiene mejores resultados con vector caracteristico mas pequeño (32)

## MLP
Input.txt format
```
// Mode
{train_load} {test} {save}

// Architectures
{depth} {input_size}
{node_size} {activation_function} //FOR EACH LAYER
{loss_function}

// If train mode (train_load=1)
{filename_x_train} {filename_y_train}
{epochs} {alpha}
// Else load mode (train_load=0)
{filename_import_model}

// If test (test=1)
{filename_x_test}

// If save (save=1)
{filename_export_model}
```

