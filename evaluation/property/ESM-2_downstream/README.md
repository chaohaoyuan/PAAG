# Modify the config files
Modify the following options in all the .yaml files "./config/downstream_task/PretrainESM2"
- dataset.path: the path to download and save data
- task.model.path: the path of model config

Modify line 176 in "./protst/model.py" to the path of parameter weight file of projector(MLP) (.pt file)

# Finetune and evaluate on fitness task
```
bash ./scripts/fitness_finetune.sh <encoder_path> <dataset> <batch_size> <fixed> <output_path> <lr> <dropout> <weight_decay>
```
- \<encoder_path\>: the path of backbone parameter weight file (.pth file)

- \<dataset\>: Task to fintune and evaluate, "BetaLactamase / Fluorescence / Stability / td_datasets.AAV / td_datasets.Thermostability"

- \<batch_size\>: batch size

- \<fixed\>: Fixed the backbone parameters or not, "True / False"

- \<output_path\>: The path of output folder

- \<lr\>: learning rate

- \<dropout\>: dropout rate

- \<weight_decay\>: weight decay

The finetune will run on 4 gpus.

# Finetune and evaluate on fitness thermo task
```
bash ./scripts/fitness_thermo_finetune.sh <encoder_path> <batch_size> <fixed> <output_path> <lr> <dropout> <weight_decay>
```
- \<encoder_path\>: the path of backbone parameter weight file (.pth file)

- \<batch_size\>: batch size

- \<fixed\>: Fixed the backbone parameters or not, "True / False"

- \<output_path\>: The path of output folder

- \<lr\>: learning rate

- \<dropout\>: dropout rate

- \<weight_decay\>: weight decay

The finetune will run on 4 gpus.

# Finetune and evaluate on localization task
```
bash ./scripts/localization_finetune.sh <encoder_path> <dataset> <batch_size> <fixed> <output_path> <lr> <dropout> <weight_decay>
```
- \<encoder_path\>: the path of backbone parameter weight file (.pth file)

- \<dataset\>: Task to fintune and evaluate, "BinaryLocalization / SubcellularLocalization"

- \<batch_size\>: batch size

- \<fixed\>: Fixed the backbone parameters or not, "True / False"

- \<output_path\>: The path of output folder

- \<lr\>: learning rate

- \<dropout\>: dropout rate

- \<weight_decay\>: weight decay

The finetune will run on 4 gpus.