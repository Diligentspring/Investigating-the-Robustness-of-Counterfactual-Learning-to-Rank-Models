# Investigating the Robustness of Counterfactual Learning to Rank Models

This repository contains the supporting code for our paper "Investigating the Robustness of Counterfactual Learning to Rank Models".

[comment]: <> (## Online Appendix)

[comment]: <> (### Figure_1 The spider chart on MSLR)

[comment]: <> (![image]&#40;Figure_1.png&#41;)

[comment]: <> (### Table_1 The ANOVA results on MSLR)

[comment]: <> (![image]&#40;Table_1.png&#41;)

[comment]: <> (### Figure_2 The effect of production ranker on CLTR models on MSLR)

[comment]: <> (![image]&#40;Figure_2.png&#41;)

[comment]: <> (### Table_2 The effect of user simulation model on CLTR models on MSLR)

[comment]: <> (![image]&#40;Table_2.png&#41;)

## Repository Usage
### Get Started
```
pip install -r requirements.txt
```

### To preprocess data
To normalize the features:
```
python extrac_feature_statistics.py [DATASET_PATH]
python normaliza_feature.py [FEATEURE_STATISTIC_FILE] [DATA_PATH] [OUTPUT_PATH] ["log"(optional)]
```

To convert the data to ULTRA form:
```
python convert_to_ULTRA.py [DATA_PATH] [feature_lenth] [OUTPUT_PATH]
```

### To utilize Plackett-Luce model to randomly generate initial ranking lists based on the relevance scores given by the ranker (e.g. 1% DNN)
```
python generate_init_list.py [ULTRA_DATA_PATH] [CLASSIFY_PATH] [weight] [OUTPUT_PATH]
```

### To generate click data for initial ranking lists
```
python generate_PBM.py [INITLIST_PATH] [eta] [ULTRA_DATA_PATH]
python generate_DCM.py [INITLIST_PATH] [beta] [eta] [ULTRA_DATA_PATH]
python generate_CBCM.py [INITLIST_PATH] [w] [g] [eta] [ULTRA_DATA_PATH]
```

### To train and evaluate CLTR models
This project reproduces the counterfactual learning to rank models based on [ULTRA_pytorch](https://github.com/ULTR-Community/ULTRA_pytorch). Here we give the simple instructions to train a CLTR model. Please see [ULTRA_pytorch](https://github.com/ULTR-Community/ULTRA_pytorch) for more details about this framework.

For example, to train a naive model (click baseline):
```
python main.py --CLTR_model NaiveAlgorithm --data_dir ./tests/data/ --train_data_prefix train --model_dir ./tests/model/clicksoftmax_256_0.1 --setting_file ./tests/test_settings/naive.json --batch_size 256 --ln 0.1 --seed 1 --max_train_iteration 10000
```
To train a DLA-PBM model:
```
python main.py --CLTR_model DLA_PBM --data_dir ./tests/data/ --train_data_prefix train --model_dir ./tests/model/DLAPBM_256_0.1 --setting_file ./tests/test_settings/dla.json --batch_size 256 --ln 0.1 --seed 1 --max_train_iteration 10000
```
To train an IPS-PBM model: First, estimate the \eta parameters of the PBM click model (either using EM algorithm or via a separate randomization experiment) and store them into a propensity file. In our paper, we used the [Pyclick](https://github.com/markovi/PyClick) library to train a PBM click model via EM algorithm. Then, run the following code:
```
python main.py --CLTR_model IPS_PBM --data_dir ./tests/data/ --train_data_prefix train --model_dir ./tests/model/IPSPBM_256_0.1 --setting_file ./tests/test_settings/ips.json --propensity_file_path ./tests/data/train/propensity.txt --batch_size 256 --ln 0.1 --seed 1 --max_train_iteration 10000
```
To evaluate a CLTR model:
```
python main.py --data_dir ./tests/data/ --setting_file ./tests/test_settings/dla.json --batch_size 256 --test_only True  --model_dir ./tests/data/model/DLAPBM_256_0.1/ultra.learning_algorithm.DLA.ckpt200
```