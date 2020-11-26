python evaluate_with_attacks.py --attack attacks.MiddleCrop --model_path models/worst_mse/checkpoints/tf_model_035_0.30_0.92.hdf5
python evaluate_with_attacks.py --attack attacks.RandomCrop --model_path models/worst_mse/checkpoints/tf_model_035_0.30_0.92.hdf5
python evaluate_with_attacks.py --attack attacks.WorstCrop --model_path models/worst_mse/checkpoints/tf_model_035_0.30_0.92.hdf5
python evaluate_with_attacks.py --attack attacks.WorstCrop --loss mse --model_path models/worst_mse/checkpoints/tf_model_035_0.30_0.92.hdf5

