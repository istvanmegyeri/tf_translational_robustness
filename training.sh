python train.py --attack attacks.MiddleCrop --save_dir models/middle/Sp1_01 --data_path data/motif_discovery/HaibH1hescSp1Pcr1xUniPk/HaibH1hescSp1Pcr1xUniPk.npz
python train.py --attack attacks.RandomCrop --save_dir models/random/Sp1_01 --data_path data/motif_discovery/HaibH1hescSp1Pcr1xUniPk/HaibH1hescSp1Pcr1xUniPk.npz
python train.py --attack attacks.WorstCrop --save_dir models/worst/Sp1_01 --data_path data/motif_discovery/HaibH1hescSp1Pcr1xUniPk/HaibH1hescSp1Pcr1xUniPk.npz
python train.py --attack attacks.WorstCrop --save_dir models/worst_mse/Sp1_01 --data_path data/motif_discovery/HaibH1hescSp1Pcr1xUniPk/HaibH1hescSp1Pcr1xUniPk.npz --loss mse
echo