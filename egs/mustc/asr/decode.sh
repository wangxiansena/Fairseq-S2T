#! /bin/bash

gpu_num=1

data_dir=
test_subset=(dev tst-COMMON)

exp_name=
if [ "$#" -eq 1 ]; then
    exp_name=$1
fi

cer=0
ctc_infer=0
n_average=10
beam_size=5
len_penalty=1.0
max_tokens=80000
dec_model=checkpoint_best.pt

cmd="./run.sh
    --stage 2
    --stop_stage 2
    --gpu_num ${gpu_num}
    --exp_name ${exp_name}
    --n_average ${n_average}
    --cer ${cer}
    --ctc_infer ${ctc_infer}
    --beam_size ${beam_size}
    --len_penalty ${len_penalty}
    --max_tokens ${max_tokens}
    --dec_model ${dec_model}
    "

if [[ -n ${data_dir} ]]; then
    cmd="$cmd --data_dir ${data_dir}"
fi
if [[ ${#test_subset[@]} -ne 0 ]]; then
    subsets=$(echo ${test_subset[*]} | sed 's/ /,/g')
    cmd="$cmd --test_subset ${subsets}"
fi

echo $cmd
eval $cmd
