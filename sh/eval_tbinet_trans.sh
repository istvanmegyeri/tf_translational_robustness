function eval_ckpts() {
  for a in $2; do
    for m in $(ls $1*.hdf5); do
#      echo $m
      sl=$(echo $m | awk -F'/' '{print $(NF-1)}' | cut -d '_' -f 3)
      echo "python eval_trans_attack.py --data_path data/encode_deepsea/valid_e.npz --attack attacks."$a" --seq_length "$sl" --loss bce --model_path "$m" --metric \"auc,aupr,bce\" --out_fname "$1""$a".csv --batch_size 1600 --attack_batch 1600 --n_try 20 --gpu "$3
    done
  done
}

function run() {
  model_dir=$1
  best_epoch=$(cat $model_dir"/metrics.csv" | dos2unix | awk -F',' 'NR >1{ print $3" "($1+1)}' | sort -g -k 1,1 -t ' ' | head -n 1 | cut -d' ' -f 2)
  if [ $best_epoch -lt 10 ]; then
    best_epoch="0"$best_epoch
  fi
  m=$model_dir"/tbinet_"$best_epoch".hdf5"
  sl=$(echo $m | awk -F'/' '{print $(NF-1)}' | cut -d '_' -f 3)
  for a in $2; do
    echo "python eval_trans_attack.py --data_path data/encode_deepsea/test_e.npz --gpu "$3" --attack attacks."$a" --seq_length "$sl" --loss bce --model_path "$m" --metric \"auc,aupr,bce\" --out_fname results/tbinet_translate.csv --batch_size 1600 --attack_batch 1600 --n_try 20"
  done
}

#eval_ckpts saved_models/tbinet_N_500_MiddleCrop/
#run saved_models/tbinet_N_1000_MiddleCrop "MiddleCrop RandomCrop WorstCrop" 3
eval_ckpts saved_models/tbinet_N_1000_WorstCrop/ "MiddleCrop RandomCrop WorstCrop" 0
run saved_models/tbinet_N_1000_WorstCrop "MiddleCrop RandomCrop WorstCrop" 0
#run saved_models/tbinet_N_500_WorstCrop
