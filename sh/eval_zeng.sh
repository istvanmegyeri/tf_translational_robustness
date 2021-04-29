out_dir=saved_models
out_fname=results/zeng/translate_val_loss.csv
evalbest(){
  #m_dirs=`ls -d $out_dir/*/*/*/*/*/`
  m_dirs=`ls -d $1`
  #neg_mode=`ls -d $out_dir/*/*/*/*/ | cut -d "/" -f 4 | sort -u`
  neg_mode=`ls -d $1 | cut -d "/" -f 4 | sort -u`
  for m_dir in $m_dirs; do
    b_path=`echo $m_dir | cut -d "/" -f 1-3`
    trans=`echo $m_dir | cut -d "/" -f 5`
    sl=`echo $m_dir | cut -d "/" -f 3`
    #val_acc
    if [ "$2" = "val_acc" ]; then
      best_epoch=`cat $m_dir"/logs.csv" | awk -F',' 'NR>1{print $4" "(NR-2)}' | sort -g -k 1,1 -t ' ' | tail -n 1 | awk '{print $2}'`
    elif [ "$2" = "val_loss" ]; then
      best_epoch=`cat $m_dir"/logs.csv" | awk -F',' 'NR>1{print $3" "(NR-2)}' | sort -g -k 1,1 -t ' ' | head -n 1 | awk '{print $2}'`
    fi
    if [[ "$best_epoch" -lt 10 ]]; then
      best_epoch="00"$best_epoch;
    elif [[ "$best_epoch" -lt 100 ]]; then
      best_epoch="0"$best_epoch;
    fi
    m_name=`ls $m_dir"checkpoints/" | grep "tf_model_"$best_epoch`
    m_path=$m_dir"checkpoints/"$m_name
    for m in $neg_mode; do
      for a in $3;do
        echo "python eval_trans_attack.py --set test --data_path data/encode_cnn_arch/"$m"/"$trans"/"$trans".npz --attack attacks."$a" --seq_length "$sl" --loss xe --metric acc --out_fname "$out_fname" --model_path "$m_path
      done
    done
  done
}
#evalbest $out_dir"/*/75/*/*/*/" "val_loss" "MiddleCrop RandomCrop WorstCrop"
#evalbest $out_dir"/*/90/*/*/*/" "val_loss" "MiddleCrop RandomCrop WorstCrop"
#evalbest $out_dir"/*/95/*/*/*/" "val_loss" "MiddleCrop RandomCrop WorstCrop"
out_fname=results/zeng/101_val_loss.csv
evalbest $out_dir"/*/101/*/*/*/" "val_loss" "MiddleCrop"