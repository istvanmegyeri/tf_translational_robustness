function eval_ckpts() {
#  for a in MiddleCrop RandomCrop WorstCrop; do
  for a in RandomCrop; do
    for m in $(ls $1*.hdf5); do
      echo $m
      sl=$(echo $m | awk -F'/' '{print $(NF-1)}' | cut -d '_' -f 3)
      python eval_trans_attack.py --data_path data/encode_deepsea/valid_e.npz --gpu 0 --attack attacks.$a --seq_length $sl --loss bce --model_path $m --metric "auc,aupr,bce" --out_fname $1""$a".csv" --batch_size 1600 --attack_batch 1600 --n_try 20
    done
  done
}

function run() {
  model_dir=$1
  best_epoch=$(cat $model_dir"/metrics.csv" | dos2unix | awk -F',' 'NR >1{ print $3" "$1}' | sort -g -k 1,1 -t ' ' | head -n 1 | cut -d' ' -f 2)
  echo $best_epoch

  #  cat $model_dir"/metrics.csv" | dos2unix | sort -g -k 3,3 -t ','
}

#eval_ckpts saved_models/tbinet_N_500_MiddleCrop/
#eval_ckpts saved_models/tbinet_N_500_RandomCrop/
#eval_ckpts saved_models/tbinet_N_500_WorstCrop/
eval_ckpts saved_models/tbinet_Y_500_MiddleCrop/

#run saved_models/tbinet_N_500_MiddleCrop
#run saved_models/tbinet_N_500_RandomCrop
#run saved_models/tbinet_N_500_WorstCrop
