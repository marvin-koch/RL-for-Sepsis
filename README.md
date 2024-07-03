# RL-for-Sepsis

Research in Data Science, ETH Zürich

## Installation
We use Python 3.9 and provide the exact environment used:

```bash
pip install -r requirements.txt
```

Important: Before executing the following steps, you will need to get access to and setup the MIMIC-III dataset: https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iii/buildmimic/postgres

## Data preprocessing

First, to extract features, run:
```bash
python3 preprocess.py
```
Then to create the patient cohort run:
```bash
python3 sepsis_cohort.py
```

Finally, to create the training, validation and test set, run:
```bash
python3 split_sepsis_cohort.py
```
This will create the following files: train_set_tuples, val_set_tuples, test_set_tuples


## Running the model

Before running the model, the file paths must be defined manually in the config file (configs/common_continous.yaml).

```bash
python3 run_model.py --model <model_type> --task <task_name> --path <path_to_model> --loss_param <loss_hyperparameters> -reward <reward_type> --reward_param <reward_hyperparameters> --device <device>
```
* --model: The model to run, can choose from <lstm, transformer>
* --task: Choose from <train, eval, eval_multiple> (Note that eval_multiple can only evaluate 4 policies and the clinician at once)
* --path: Path to model file (.pth). For LSTM models, the file must end with "\_{number of nodes per layer}.pth" (e.g. lstm_16.pth). For Transformers, the file must end with "{layers}\_{heads}.pth" (e.g. transformer_1_1.pth). Multiple models can be passed by seperating the file names with "," (without spaces). The path must be defined to be able to set the number of nodes/layers/heads.
* --loss_param (Optional):  Define hyparameters λ_1 and λ_2 in the loss, the parameters should be comma seperated without spaces (e.g. 1,1). If the loss isn't defined or "none" is passed, then the parameters are equal to 1, 1.
* --reward (Optional): Define bias in the SOFA score used in the reward function, by passing the type of bias <simple, both> and the subscore which we would like to be biased towards, remember that there are only 6 subscores. The arguments should be comma seperated without spaces (e.g. simple,3) If the reward isn't defined or "none" is passed, then we use the original SOFA score.
* --reward_param (Optional): Define hyparameters γ_1 and γ_2 in the reward, the parameters should be comma seperated without spaces (e.g. 0.125,0.2). If the parameters aren't defined or "none" is passed, then the parameters are equal to 0.125, 0.2.
* --device (Optional): Choose from <cpu, mps, cuda>. The default value is cpu.


Example
```bash
python3 run_model.py --model lstm --task train --path lstm_16.pth --reward both,2 --device cuda
```
## Reproducing the results
To reproduce our results, we give the following bash script (note that you have to run the preprocessing steps before):
```bash
sh main.sh <model> <mode> <device>
```
* model: The model to run, can choose from <lstm, transformer>
* mode: Which configuration you wish to reproduce, can choose from <normal, bias_loss, reward_simple, reward_both, reward_param> (Refer to the report for more explanations)
* device: Choose from <cpu, mps, cuda>.


## References
* The preprocessing steps are forked from: https://github.com/microsoft/mimic_sepsis.
* The environment was forked from: https://github.com/acneyouth1996/RL-for-sepsis-continuous/blob/yong_v0/
* The TD3 implementation with LSTM encoder was forked from: https://github.com/twni2016/pomdp-baselines/blob/main/
* The TD3 implementation with GPT-2 encoder was forked from: https://github.com/twni2016/Memory-RL

