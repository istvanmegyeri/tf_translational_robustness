# tf_attack

## Evironment setup
```
conda create -n env_name python=3.6
conda activate env_name
conda install tensorflow-gpu==2.1
```
## Datasets
* Zeng's dataset
* DeepSea dataset

Data conversion from `.mat` to `.npz` can be done by:  
`python convert_data.py --in_fname path_to_mat --out_fname path_to_output_npz`

Alternatively, you can download our converted data from [here](one_drive_link). 

## Model training
Our best models can be downloaded from [here](add_link_here). Checkpoints are also avaiable [here](add_link_here).

### Normal training
* on Zeng's dataset:
```
python train_zeng.py --attack attacks.MiddleCrop  
python train_zeng.py --attack attacks.RandomCrop
```
* on DeepSea dataset:
```
...
```
### Adversarial training
* on Zeng's dataset:  
```
python train_zeng.py --attack attacks.WorstCrop
```
* on DeepSea dataset:
```
...
```

## Model evaluation
Under translation attack:
```
python eval_trans_attack.p --attack attacks.WorstCrop --model_path path_to_model
```
Under ood attack:
```
python eval_ood_attack.p --attack attacks.Shuffle --data_path path_to_test_set --model_path path_to_model
python eval_ood_attack.p --attack attacks.Uniform --data_path path_to_test_set --model_path path_to_model
```