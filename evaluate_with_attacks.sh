python evaluate_with_attacks.py --attack attacks.MiddleCrop --model_path models/middle/Mafk_01/checkpoints/tf_model_039_0.26_0.94.hdf5 --data_path data/motif_occupancy/SydhImr90MafkIggrabUniPk/SydhImr90MafkIggrabUniPk.npz
python evaluate_with_attacks.py --attack attacks.RandomCrop --model_path models/middle/Mafk_01/checkpoints/tf_model_039_0.26_0.94.hdf5 --data_path data/motif_occupancy/SydhImr90MafkIggrabUniPk/SydhImr90MafkIggrabUniPk.npz
python evaluate_with_attacks.py --attack attacks.WorstCrop --model_path models/middle/Mafk_01/checkpoints/tf_model_039_0.26_0.94.hdf5 --data_path data/motif_occupancy/SydhImr90MafkIggrabUniPk/SydhImr90MafkIggrabUniPk.npz
python evaluate_with_attacks.py --attack attacks.WorstCrop --model_path models/middle/Mafk_01/checkpoints/tf_model_039_0.26_0.94.hdf5 --data_path data/motif_occupancy/SydhImr90MafkIggrabUniPk/SydhImr90MafkIggrabUniPk.npz --loss xe
echo
python evaluate_with_attacks.py --attack attacks.MiddleCrop --model_path models/random/Mafk_01/checkpoints/tf_model_038_0.26_0.94.hdf5 --data_path data/motif_discovery/SydhImr90MafkIggrabUniPk/SydhImr90MafkIggrabUniPk.npz
python evaluate_with_attacks.py --attack attacks.RandomCrop --model_path models/random/Mafk_01/checkpoints/tf_model_038_0.26_0.94.hdf5 --data_path data/motif_discovery/SydhImr90MafkIggrabUniPk/SydhImr90MafkIggrabUniPk.npz
python evaluate_with_attacks.py --attack attacks.WorstCrop --model_path models/random/Mafk_01/checkpoints/tf_model_038_0.26_0.94.hdf5 --data_path data/motif_discovery/SydhImr90MafkIggrabUniPk/SydhImr90MafkIggrabUniPk.npz
python evaluate_with_attacks.py --attack attacks.WorstCrop --model_path models/random/Mafk_01/checkpoints/tf_model_038_0.26_0.94.hdf5 --data_path data/motif_discovery/SydhImr90MafkIggrabUniPk/SydhImr90MafkIggrabUniPk.npz --loss xe

python evaluate_with_attacks.py --attack attacks.MiddleCrop --model_path models/worst/Mafk_01/checkpoints/tf_model_010_0.27_0.92.hdf5 --data_path data/motif_discovery/SydhImr90MafkIggrabUniPk/SydhImr90MafkIggrabUniPk.npz
python evaluate_with_attacks.py --attack attacks.RandomCrop --model_path models/worst/Mafk_01/checkpoints/tf_model_010_0.27_0.92.hdf5 --data_path data/motif_discovery/SydhImr90MafkIggrabUniPk/SydhImr90MafkIggrabUniPk.npz
python evaluate_with_attacks.py --attack attacks.WorstCrop --model_path models/worst/Mafk_01/checkpoints/tf_model_010_0.27_0.92.hdf5 --data_path data/motif_discovery/SydhImr90MafkIggrabUniPk/SydhImr90MafkIggrabUniPk.npz
python evaluate_with_attacks.py --attack attacks.WorstCrop --model_path models/worst/Mafk_01/checkpoints/tf_model_010_0.27_0.92.hdf5 --data_path data/motif_discovery/SydhImr90MafkIggrabUniPk/SydhImr90MafkIggrabUniPk.npz --loss xe

python evaluate_with_attacks.py --attack attacks.MiddleCrop --model_path models/worst_mse/Mafk_01/checkpoints/tf_model_037_0.30_0.92.hdf5 --data_path data/motif_discovery/SydhImr90MafkIggrabUniPk/SydhImr90MafkIggrabUniPk.npz
python evaluate_with_attacks.py --attack attacks.RandomCrop --model_path models/worst_mse/Mafk_01/checkpoints/tf_model_037_0.30_0.92.hdf5 --data_path data/motif_discovery/SydhImr90MafkIggrabUniPk/SydhImr90MafkIggrabUniPk.npz
python evaluate_with_attacks.py --attack attacks.WorstCrop --model_path models/worst_mse/Mafk_01/checkpoints/tf_model_037_0.30_0.92.hdf5 --data_path data/motif_discovery/SydhImr90MafkIggrabUniPk/SydhImr90MafkIggrabUniPk.npz
python evaluate_with_attacks.py --attack attacks.WorstCrop --model_path models/worst_mse/Mafk_01/checkpoints/tf_model_037_0.30_0.92.hdf5 --data_path data/motif_discovery/SydhImr90MafkIggrabUniPk/SydhImr90MafkIggrabUniPk.npz --loss xe


python evaluate_with_attacks.py --attack attacks.MiddleCrop --model_path models/middle/Sp1_01/checkpoints/tf_model_042_0.62_0.75.hdf5 --data_path data/motif_discovery/HaibH1hescSp1Pcr1xUniPk/HaibH1hescSp1Pcr1xUniPk.npz
python evaluate_with_attacks.py --attack attacks.RandomCrop --model_path models/middle/Sp1_01/checkpoints/tf_model_042_0.62_0.75.hdf5 --data_path data/motif_discovery/HaibH1hescSp1Pcr1xUniPk/HaibH1hescSp1Pcr1xUniPk.npz
python evaluate_with_attacks.py --attack attacks.WorstCrop --model_path models/middle/Sp1_01/checkpoints/tf_model_042_0.62_0.75.hdf5 --data_path data/motif_discovery/HaibH1hescSp1Pcr1xUniPk/HaibH1hescSp1Pcr1xUniPk.npz
python evaluate_with_attacks.py --attack attacks.WorstCrop --model_path models/middle/Sp1_01/checkpoints/tf_model_042_0.62_0.75.hdf5 --data_path data/motif_discovery/HaibH1hescSp1Pcr1xUniPk/HaibH1hescSp1Pcr1xUniPk.npz --loss xe

python evaluate_with_attacks.py --attack attacks.MiddleCrop --model_path models/random/Sp1_01/checkpoints/tf_model_025_0.61_0.76.hdf5 --data_path data/motif_discovery/HaibH1hescSp1Pcr1xUniPk/HaibH1hescSp1Pcr1xUniPk.npz
python evaluate_with_attacks.py --attack attacks.RandomCrop --model_path models/random/Sp1_01/checkpoints/tf_model_025_0.61_0.76.hdf5 --data_path data/motif_discovery/HaibH1hescSp1Pcr1xUniPk/HaibH1hescSp1Pcr1xUniPk.npz
python evaluate_with_attacks.py --attack attacks.WorstCrop --model_path models/random/Sp1_01/checkpoints/tf_model_025_0.61_0.76.hdf5 --data_path data/motif_discovery/HaibH1hescSp1Pcr1xUniPk/HaibH1hescSp1Pcr1xUniPk.npz
python evaluate_with_attacks.py --attack attacks.WorstCrop --model_path models/random/Sp1_01/checkpoints/tf_model_025_0.61_0.76.hdf5 --data_path data/motif_discovery/HaibH1hescSp1Pcr1xUniPk/HaibH1hescSp1Pcr1xUniPk.npz --loss xe

python evaluate_with_attacks.py --attack attacks.MiddleCrop --model_path models/worst/Sp1_01/checkpoints/tf_model_022_0.63_0.67.hdf5 --data_path data/motif_discovery/HaibH1hescSp1Pcr1xUniPk/HaibH1hescSp1Pcr1xUniPk.npz
python evaluate_with_attacks.py --attack attacks.RandomCrop --model_path models/worst/Sp1_01/checkpoints/tf_model_022_0.63_0.67.hdf5 --data_path data/motif_discovery/HaibH1hescSp1Pcr1xUniPk/HaibH1hescSp1Pcr1xUniPk.npz
python evaluate_with_attacks.py --attack attacks.WorstCrop --model_path models/worst/Sp1_01/checkpoints/tf_model_022_0.63_0.67.hdf5 --data_path data/motif_discovery/HaibH1hescSp1Pcr1xUniPk/HaibH1hescSp1Pcr1xUniPk.npz
python evaluate_with_attacks.py --attack attacks.WorstCrop --model_path models/worst/Sp1_01/checkpoints/tf_model_022_0.63_0.67.hdf5 --data_path data/motif_discovery/HaibH1hescSp1Pcr1xUniPk/HaibH1hescSp1Pcr1xUniPk.npz --loss xe

python evaluate_with_attacks.py --attack attacks.MiddleCrop --model_path models/worst_mse/Sp1_01/checkpoints/tf_model_006_0.72_0.53.hdf5 --data_path data/motif_discovery/HaibH1hescSp1Pcr1xUniPk/HaibH1hescSp1Pcr1xUniPk.npz
python evaluate_with_attacks.py --attack attacks.RandomCrop --model_path models/worst_mse/Sp1_01/checkpoints/tf_model_006_0.72_0.53.hdf5 --data_path data/motif_discovery/HaibH1hescSp1Pcr1xUniPk/HaibH1hescSp1Pcr1xUniPk.npz
python evaluate_with_attacks.py --attack attacks.WorstCrop --model_path models/worst_mse/Sp1_01/checkpoints/tf_model_006_0.72_0.53.hdf5 --data_path data/motif_discovery/HaibH1hescSp1Pcr1xUniPk/HaibH1hescSp1Pcr1xUniPk.npz
python evaluate_with_attacks.py --attack attacks.WorstCrop --model_path models/worst_mse/Sp1_01/checkpoints/tf_model_006_0.72_0.53.hdf5 --data_path data/motif_discovery/HaibH1hescSp1Pcr1xUniPk/HaibH1hescSp1Pcr1xUniPk.npz --loss xe


python evaluate_with_attacks.py --attack attacks.MiddleCrop --model_path models/middle/Znf_01/checkpoints/tf_model_038_0.45_0.87.hdf5 --data_path data/motif_discovery/SydhK562Znf143IggrabUniPk/SydhK562Znf143IggrabUniPk.npz
python evaluate_with_attacks.py --attack attacks.RandomCrop --model_path models/middle/Znf_01/checkpoints/tf_model_038_0.45_0.87.hdf5 --data_path data/motif_discovery/SydhK562Znf143IggrabUniPk/SydhK562Znf143IggrabUniPk.npz
python evaluate_with_attacks.py --attack attacks.WorstCrop --model_path models/middle/Znf_01/checkpoints/tf_model_038_0.45_0.87.hdf5 --data_path data/motif_discovery/SydhK562Znf143IggrabUniPk/SydhK562Znf143IggrabUniPk.npz
python evaluate_with_attacks.py --attack attacks.WorstCrop --model_path models/middle/Znf_01/checkpoints/tf_model_038_0.45_0.87.hdf5 --data_path data/motif_discovery/SydhK562Znf143IggrabUniPk/SydhK562Znf143IggrabUniPk.npz --loss xe

python evaluate_with_attacks.py --attack attacks.MiddleCrop --model_path models/random/Znf_01/checkpoints/tf_model_040_0.43_0.87.hdf5 --data_path data/motif_discovery/SydhK562Znf143IggrabUniPk/SydhK562Znf143IggrabUniPk.npz
python evaluate_with_attacks.py --attack attacks.RandomCrop --model_path models/random/Znf_01/checkpoints/tf_model_040_0.43_0.87.hdf5 --data_path data/motif_discovery/SydhK562Znf143IggrabUniPk/SydhK562Znf143IggrabUniPk.npz
python evaluate_with_attacks.py --attack attacks.WorstCrop --model_path models/random/Znf_01/checkpoints/tf_model_040_0.43_0.87.hdf5 --data_path data/motif_discovery/SydhK562Znf143IggrabUniPk/SydhK562Znf143IggrabUniPk.npz
python evaluate_with_attacks.py --attack attacks.WorstCrop --model_path models/random/Znf_01/checkpoints/tf_model_040_0.43_0.87.hdf5 --data_path data/motif_discovery/SydhK562Znf143IggrabUniPk/SydhK562Znf143IggrabUniPk.npz --loss xe

python evaluate_with_attacks.py --attack attacks.MiddleCrop --model_path models/worst/Znf_01/checkpoints/tf_model_011_0.42_0.84.hdf5 --data_path data/motif_discovery/SydhK562Znf143IggrabUniPk/SydhK562Znf143IggrabUniPk.npz
python evaluate_with_attacks.py --attack attacks.RandomCrop --model_path models/worst/Znf_01/checkpoints/tf_model_011_0.42_0.84.hdf5 --data_path data/motif_discovery/SydhK562Znf143IggrabUniPk/SydhK562Znf143IggrabUniPk.npz
python evaluate_with_attacks.py --attack attacks.WorstCrop --model_path models/worst/Znf_01/checkpoints/tf_model_011_0.42_0.84.hdf5 --data_path data/motif_discovery/SydhK562Znf143IggrabUniPk/SydhK562Znf143IggrabUniPk.npz
python evaluate_with_attacks.py --attack attacks.WorstCrop --model_path models/worst/Znf_01/checkpoints/tf_model_011_0.42_0.84.hdf5 --data_path data/motif_discovery/SydhK562Znf143IggrabUniPk/SydhK562Znf143IggrabUniPk.npz --loss xe

python evaluate_with_attacks.py --attack attacks.MiddleCrop --model_path models/worst_mse/Znf_01/checkpoints/tf_model_028_0.46_0.85.hdf5 --data_path data/motif_discovery/SydhK562Znf143IggrabUniPk/SydhK562Znf143IggrabUniPk.npz
python evaluate_with_attacks.py --attack attacks.RandomCrop --model_path models/worst_mse/Znf_01/checkpoints/tf_model_028_0.46_0.85.hdf5 --data_path data/motif_discovery/SydhK562Znf143IggrabUniPk/SydhK562Znf143IggrabUniPk.npz
python evaluate_with_attacks.py --attack attacks.WorstCrop --model_path models/worst_mse/Znf_01/checkpoints/tf_model_028_0.46_0.85.hdf5 --data_path data/motif_discovery/SydhK562Znf143IggrabUniPk/SydhK562Znf143IggrabUniPk.npz
python evaluate_with_attacks.py --attack attacks.WorstCrop --model_path models/worst_mse/Znf_01/checkpoints/tf_model_028_0.46_0.85.hdf5 --data_path data/motif_discovery/SydhK562Znf143IggrabUniPk/SydhK562Znf143IggrabUniPk.npz --loss xe


python evaluate_with_attacks.py --attack attacks.MiddleCrop --model_path models/middle/Mafk_01_oc/checkpoints/tf_model_016_0.59_0.74.hdf5 --data_path data/motif_occupancy/SydhImr90MafkIggrabUniPk/SydhImr90MafkIggrabUniPk.npz
python evaluate_with_attacks.py --attack attacks.RandomCrop --model_path models/middle/Mafk_01_oc/checkpoints/tf_model_016_0.59_0.74.hdf5 --data_path data/motif_occupancy/SydhImr90MafkIggrabUniPk/SydhImr90MafkIggrabUniPk.npz
python evaluate_with_attacks.py --attack attacks.WorstCrop --model_path models/middle/Mafk_01_oc/checkpoints/tf_model_016_0.59_0.74.hdf5 --data_path data/motif_occupancy/SydhImr90MafkIggrabUniPk/SydhImr90MafkIggrabUniPk.npz
python evaluate_with_attacks.py --attack attacks.WorstCrop --model_path models/middle/Mafk_01_oc/checkpoints/tf_model_016_0.59_0.74.hdf5 --data_path data/motif_occupancy/SydhImr90MafkIggrabUniPk/SydhImr90MafkIggrabUniPk.npz --loss xe

python evaluate_with_attacks.py --attack attacks.MiddleCrop --model_path models/random/Mafk_01_oc/checkpoints/tf_model_035_0.59_0.74.hdf5 --data_path data/motif_occupancy/SydhImr90MafkIggrabUniPk/SydhImr90MafkIggrabUniPk.npz
python evaluate_with_attacks.py --attack attacks.RandomCrop --model_path models/random/Mafk_01_oc/checkpoints/tf_model_035_0.59_0.74.hdf5 --data_path data/motif_occupancy/SydhImr90MafkIggrabUniPk/SydhImr90MafkIggrabUniPk.npz
python evaluate_with_attacks.py --attack attacks.WorstCrop --model_path models/random/Mafk_01_oc/checkpoints/tf_model_035_0.59_0.74.hdf5 --data_path data/motif_occupancy/SydhImr90MafkIggrabUniPk/SydhImr90MafkIggrabUniPk.npz
python evaluate_with_attacks.py --attack attacks.WorstCrop --model_path models/random/Mafk_01_oc/checkpoints/tf_model_035_0.59_0.74.hdf5 --data_path data/motif_occupancy/SydhImr90MafkIggrabUniPk/SydhImr90MafkIggrabUniPk.npz --loss xe

python evaluate_with_attacks.py --attack attacks.MiddleCrop --model_path models/worst/Mafk_01_oc/checkpoints/tf_model_006_0.59_0.70.hdf5 --data_path data/motif_occupancy/SydhImr90MafkIggrabUniPk/SydhImr90MafkIggrabUniPk.npz
python evaluate_with_attacks.py --attack attacks.RandomCrop --model_path models/worst/Mafk_01_oc/checkpoints/tf_model_006_0.59_0.70.hdf5 --data_path data/motif_occupancy/SydhImr90MafkIggrabUniPk/SydhImr90MafkIggrabUniPk.npz
python evaluate_with_attacks.py --attack attacks.WorstCrop --model_path models/worst/Mafk_01_oc/checkpoints/tf_model_006_0.59_0.70.hdf5 --data_path data/motif_occupancy/SydhImr90MafkIggrabUniPk/SydhImr90MafkIggrabUniPk.npz
python evaluate_with_attacks.py --attack attacks.WorstCrop --model_path models/worst/Mafk_01_oc/checkpoints/tf_model_006_0.59_0.70.hdf5 --data_path data/motif_occupancy/SydhImr90MafkIggrabUniPk/SydhImr90MafkIggrabUniPk.npz --loss xe

python evaluate_with_attacks.py --attack attacks.MiddleCrop --model_path models/worst_mse/Mafk_01_oc/checkpoints/tf_model_016_0.58_0.72.hdf5 --data_path data/motif_occupancy/SydhImr90MafkIggrabUniPk/SydhImr90MafkIggrabUniPk.npz
python evaluate_with_attacks.py --attack attacks.RandomCrop --model_path models/worst_mse/Mafk_01_oc/checkpoints/tf_model_016_0.58_0.72.hdf5 --data_path data/motif_occupancy/SydhImr90MafkIggrabUniPk/SydhImr90MafkIggrabUniPk.npz
python evaluate_with_attacks.py --attack attacks.WorstCrop --model_path models/worst_mse/Mafk_01_oc/checkpoints/tf_model_016_0.58_0.72.hdf5 --data_path data/motif_occupancy/SydhImr90MafkIggrabUniPk/SydhImr90MafkIggrabUniPk.npz
python evaluate_with_attacks.py --attack attacks.WorstCrop --model_path models/worst_mse/Mafk_01_oc/checkpoints/tf_model_016_0.58_0.72.hdf5 --data_path data/motif_occupancy/SydhImr90MafkIggrabUniPk/SydhImr90MafkIggrabUniPk.npz --loss xe








echo

