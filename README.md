# RL-for-Sepsis

Research in Data Science Project, ETH Zürich

## References
* The preprocessing steps are forked from: https://github.com/microsoft/mimic_sepsis.
* The environment was forked from: https://github.com/acneyouth1996/RL-for-sepsis-continuous/blob/yong_v0/
* The TD3 implementation with LSTM encoder was forked from: https://github.com/twni2016/pomdp-baselines/blob/main/
* The TD3 implementation with GPT-2 encoder was forked from: https://github.com/twni2016/Memory-RL

## Installation
```bash
pip install -r requirements.txt
```
We use python 3.9

Before executing the following steps, you will need to get access to and setup the MIMIC-III dataset: https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iii/buildmimic/postgres

## Data preprocessing


First, to extract features, run:
```bash
python3 preprocess.py
```
Then to create the patient cohorts run:
```bash
python3 sepsis_cohort.py
```

Finally, to create the training, validation and test set, run:
```bash
python3 split_sepsis_cohort.py
```
This will create files train_set_tuples/val_set_tuples/test_set_tuples


## Running the model

```bash
python3 run_model.py --model <model_type> --task <task_name> --path <path_to_model> --loss <loss_bias> -reward <reward_bias> --device <device>
```
* --model: The model to run, can choose from <lstm, transformer>
* --task: Choose from <train, eval, eval_multiple>
* --path (Optional): Path to model file (.pth). For LSTM models, the file must end with the number of nodes per layer (e.g. lstm_16.pth). For Transformers, the file must end with the number of layers and heads (e.g. transformer_1_1.pth).
* --loss (Optional):  Define bias towards hyparameters λ_1 and λ_2 in the loss, the parameters should be comma seperated without spaces (e.g. 1,1). If the loss isn't definied or "none" is passed, then the parameters are equal to 1, 1.
* --reward (Optional): Define bias in the SOFA score, by simply passing the subsore which we would like to be biased towards (e.g. 3), remember that there are only 6 subscores. If the reward isn't definied or "none" is passed, then we use the original SOFA score.
* --device (Optional): Choose from <cpu, mps, cuda>. The default value is cpu.
## To replicate the results from our paper

```bash
sh main.sh
```
