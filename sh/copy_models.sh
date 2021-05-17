if [[ $# -ne 1 ]]; then
    echo "usage: ./copy_models.sh target_dir"
    exit 0
fi
tgt_dir=$1
mkdir -p $1

function mv_best_models(){
    for m_dir in `ls -d $1`; do
        echo $m_dir
        best_epoch=`cat $m_dir"/logs.csv" | awk -F',' 'NR>1{print $3" "(NR-2)}' | sort -g -k 1,1 -t ' ' | head -n 1 | awk '{print $2}'`
        if [[ "$best_epoch" -lt "10" ]]; then
            best_epoch="00"$best_epoch
        elif [[ "$best_epoch" -lt "100" ]]; then
            best_epoch="0"$best_epoch
        fi
        echo $best_epoch
        m_path=$m_dir"checkpoints/"`ls $m_dir/checkpoints/ | grep "tf_model_"$best_epoch`
        mkdir -p $tgt_dir/$m_dir
        cp $m_path $tgt_dir/$m_dir
    done
}