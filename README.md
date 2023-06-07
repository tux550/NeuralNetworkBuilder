# IA-p3

## MLP
Input.txt format
```
// Mode
{train_load} {test} {save}

// Architectures
{input_size} {output_size} {depth}
{nodes} {activation_function} {loss_function} //TODO: for each layer

// If train mode (train_load=1)
{filename_x_train} {filename_y_train}
{epochs} {alpha} {batch_size}
// Else load mode (train_load=0)
{filename_import_model}

// If test (test=1)
{filename_x_test}

// If save (save=1)
{filename_export_model}
```

