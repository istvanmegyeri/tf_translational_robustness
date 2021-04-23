out_dir="saved_models"
function train {
  for reg in preset NO; do
    for sl in 75 90 95; do
      for fn in `ls data/encode_cnn_arch/*/*/*.npz`; do
        for a in MiddleCrop RandomCrop WorstCrop; do
          #echo $fn" "$a" "$sl;
          dname=`echo $fn | cut -d "/" -f 3-4`
          echo "python train_zeng.py --fname "$fn" --attack attacks."$a" --seq_length "$sl" --save_dir "$out_dir"/"$reg"/"$sl"/"$dname"/"$a" --reg "$reg
        done
      done
    done
  done
}

function train_tbinet {
  for sl in 500 1000; do
    for add_s in N Y; do
      for a in MiddleCrop RandomCrop WorstCrop; do
        #echo $fn" "$a" "$sl;
        echo "python train_tbinet.py --data_dir data/encode_deepsea/ --add_shuf "$add_s" --attack attacks."$a" --seq_length "$sl" --save_dir "$out_dir"/tbinet_"$add_s"_"$sl"_"$a
      done
    done
  done
}

#train
train_tbinet

