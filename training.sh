python train.py --attack attacks.MiddleCrop --save_dir models/middle
python train.py --attack attacks.RandomCrop --save_dir models/random
python train.py --attack attacks.WorstCrop --save_dir models/worst
python train.py --attack attacks.WorstCrop --save_dir models/worst_mse --loss mse
