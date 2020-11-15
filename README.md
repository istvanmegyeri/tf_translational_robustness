# tf_attack

Python version: 3.8.3

Tensorflow-gpu: 2.2.0


## Model training

### Normal training
`python train.py --attack attacks.MiddleCrop`

### Random training
`python train.py --attack attacks.RandomCrop`

### Adversarial training
`python train.py --attack attacks.WorstCrop`
