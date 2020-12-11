python train.py --attack attacks.MiddleCrop --save_dir models/middle/Mafk_01_oc --data_path data/motif_occupancy/SydhImr90MafkIggrabUniPk/SydhImr90MafkIggrabUniPk.npz
python train.py --attack attacks.RandomCrop --save_dir models/random/Mafk_01_oc --data_path data/motif_occupancy/SydhImr90MafkIggrabUniPk/SydhImr90MafkIggrabUniPk.npz
python train.py --attack attacks.WorstCrop --save_dir models/worst/Mafk_01_oc --data_path data/motif_occupancy/SydhImr90MafkIggrabUniPk/SydhImr90MafkIggrabUniPk.npz
python train.py --attack attacks.WorstCrop --save_dir models/worst_mse/Mafk_01_oc --data_path data/motif_occupancy/SydhImr90MafkIggrabUniPk/SydhImr90MafkIggrabUniPk.npz --loss xe
echo

python train.py --attack attacks.WorstCrop --save_dir models/worst_mse/Sp1_02 --data_path data/motif_discovery/HaibH1hescSp1Pcr1xUniPk/HaibH1hescSp1Pcr1xUniPk.npz --loss xe
echo