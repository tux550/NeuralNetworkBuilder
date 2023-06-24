# IA-p3

## Ejecucción
1. Instalar armadillo: [Ubuntu](http://codingadventures.org/2020/05/24/how-to-install-armadillo-library-in-ubuntu/)
2. Compilar C++
   ```
   cd NeuralNetwork
   make
   ```
3. Descargar [BBDD](https://drive.google.com/file/d/1mpUk2SAyRjtzxJtQrz2owpOshF_QJAWB/view) y descomprimir en carpeta Test/raw_dataset
4. Procesar BBDD
   ```
   cd Test
   python3 initialize.py
   ```
6. Ejecutar
   - OPCION A: Enbtrenar todos los modelos y generar metricas
   ```
   cd Test
   python3 run.py
   ```
   - OPCION B: Entrenar mejor modelo
   ```
   cd Test
   python3 best.py
   ```
   - OPCION C: Importar mejor modelo
   ```
   cd Test
   python3 load.py
   ```

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

