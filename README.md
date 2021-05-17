# tf_attack

## Evironment setup
```
conda create -n env_name python=3.6
conda activate env_name
conda install tensorflow-gpu==2.1
```
## Datasets
* [Zeng's dataset](http://cnn.csail.mit.edu/)
* [DeepSea dataset information](http://deepsea.princeton.edu/help/ "DeepSea dataset")
    * [Direct download link of .mat files](http://deepsea.princeton.edu/media/code/deepsea_train_bundle.v0.9.tar.gz)

Data conversion from `.mat` to `.npz` can be done by:  
`python convert_data.py --in_fname path_to_mat --out_fname path_to_output_npz`

Alternatively, you can download our converted data from [here](https://uszeged-my.sharepoint.com/:f:/g/personal/pap_gergely_1_o365_u-szeged_hu/EmfyJP3jFWFLjxe_NK2t3N0BGFOBTn-kCO0Id8dOoV9N0A?e=yAiAx2). 

## Model training
Our best models can be downloaded from [here](https://uszeged-my.sharepoint.com/:f:/g/personal/pap_gergely_1_o365_u-szeged_hu/Eo9ntvgjGjdMjWVLWxgLXq4Bz4L9fqJCyhbM8wuX1wdLIw?e=6LDzNK).

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
python eval_trans_attack.p --attack attacks.WorstCrop --model_path path_to_model --data_path path_to_data
```
Under ood attack:
```
python eval_ood_attack.py --attack attacks.Shuffle --data_path path_to_test_set --test_path path_to_test_set --m_path path_to_model
python eval_ood_attack.py --attack attacks.Uniform --data_path path_to_test_set --test_path path_to_test_set --m_path path_to_model
python eval_ood_attack.py --attack attacks.GenSeq --data_path path_to_test_set --test_path path_to_test_set --m_path path_to_model
```

Partial (incomplete) Results of the Evaluations: [here](https://uszeged-my.sharepoint.com/:f:/g/personal/pap_gergely_1_o365_u-szeged_hu/EsIWiEKSJMZPrSRmeuspUdwBStlp_nT6SEdWzadSsDfLIQ?e=jZ8EoO)
