#!/bin/bash

set -e

if [ "$#" != 3 ]; then
    echo $2
    echo Usage. $0 expdir lang_root layers overrides
fi

expdir=$1
# lang_root=$3
layers=$2
overrides=$3
if [ ! -d "$expdir" ]; then
    echo "The expdir does not exist!"
    exit 1
fi

# if [ ! -d "$lang_root" ]; then
#     echo "VoxCeleb1 dataset does not exist!"
#     exit 1
# fi
echo "layers are ${layers}"
echo "Start testing ckpts..."
# exitxss
for state_name in 20000 40000 60000 80000 100000 120000 140000 160000 180000 200000;
do
    ckpt_path="$expdir/states-$state_name.ckpt"
    echo "Testing $ckpt_path"
    if [ ! -f "$ckpt_path" ]; then
        continue
    fi

    log_dir="$expdir/states-$state_name"
    if [ ! -d "$log_dir" ] || [ "$(cat "$log_dir"/log.txt | grep "valid-EER" | wc -l)" -lt 1 ] || [ ! -f $log_dir/valid_predict.txt ]; then
        mkdir -p $log_dir
        override="args.expdir=${log_dir},,${overrides}" 
        echo $layers
        python3 run_downstream.py -m evaluate -e $ckpt_path -o $override -l $layers -t valid > $log_dir/log.txt
    fi
done

echo "Report the testing results..."
report=$expdir/report.txt
grep valid-EER $expdir/*/log.txt | sort -nrk 2 > $report
ckpt_num=$(cat $report | wc -l)
cat $report

echo
echo "$ckpt_num checkpoints evaluated."
echo "The best checkpoint achieves EER $(cat $report | tail -n 1 | cut -d " " -f 2)"
z=$(cat $report | tail -n 1 | cut -d ":" -f 1)
best_ckpt=${z:0:-8}.ckpt
echo "Evaluating $best_ckpt"

# echo "z"
# echo "${overrides}"
# echo $overrides
# echo "z"
# python3 run_downstream.py -m evaluate -e $best_ckpt -o $overrides -l $layers -t test_known >> $expdir/log_testkn.txt
# python3 run_downstream.py -m evaluate -e $best_ckpt -o $overrides -l $layers -t test_unknown >> $expdir/log_testun.txt
python3 run_downstream.py -m evaluate -e $best_ckpt -o $overrides -l $layers -t test_known_noisy >> $expdir/log_testkn_noisy.txt
python3 run_downstream.py -m evaluate -e $best_ckpt -o $overrides -l $layers -t test_unknown_noisy >> $expdir/log_testun_noisy.txt


# echo $report

# echo
# echo "Prepare prediction file for submission..."
# best_prediction=$(realpath $(dirname $(cat $report | tail -n 1 | cut -d ":" -f 1))/$expdir/valid_eval_predict.txt)

# target=$expdir/valid_predict.txt
# if [ -f $target ]; then
#     rm $target
# fi
# target=$(realpath $target)
# ln -s $best_prediction $target

# echo "The best prediction file has been prepared"
# echo "${best_prediction} -> ${target}"
