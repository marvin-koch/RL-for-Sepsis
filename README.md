# RL-for-Sepsis

Research in Data Science Project, ETH Zürich

## References
* The preprocessing steps are forked from: https://github.com/microsoft/mimic_sepsis.
* The environment was forked from: https://github.com/acneyouth1996/RL-for-sepsis-continuous/blob/yong_v0/
* The TD3 implementation with LSTM encoder was forked from: https://github.com/twni2016/pomdp-baselines/blob/main/
* The TD3 implementation with GPT-2 encoder was forked from: https://github.com/twni2016/Memory-RL
## Data preprocessing

Before executing the following steps, you will need to get access to and setup the MIMIC-III dataset: https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iii/buildmimic/postgres

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
python3 run_model.py --model <model_type> --task <task_name> --path <path_to_model> --loss <loss> -reward <reward_bias>
```

## To replicate the results from our paper

```bash
sh main.sh
```
