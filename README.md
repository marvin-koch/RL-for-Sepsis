# RL-for-Sepsis

Research in Data Science Project, ETH Zürich

## References
The preprocessing steps are forked from: https://github.com/microsoft/mimic_sepsis.
The environment was forked from: https://github.com/acneyouth1996/RL-for-sepsis-continuous/blob/yong_v0/
The TD3 implementation with LSTM encoder was forked from: https://github.com/twni2016/pomdp-baselines/blob/main/
The TD3 implementation with GPT-2 encoder was forked from: https://github.com/twni2016/Memory-RL
## Initial data extraction and preprocess

```bash
python3 preprocess.py
```

This step extract relavent features and perform some initial preprocess, original code is from
https://github.com/microsoft/mimic_sepsis


## Create and split sepsis cohort 

```bash
python3 sepsis_cohort.py
```

and 

```bash
python3 split_sepsis_cohort.py
```

This will create files train_set_tuples/val_set_tuples/test_set_tuples


## Policy learning

```bash
python3 train_model.py
```

## To replicate the results from our paper

```bash
sh run.sh
```
