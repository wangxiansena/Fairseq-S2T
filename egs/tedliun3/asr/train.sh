#! /bin/bash

# training the model

gpu_num=2
update_freq=3
max_tokens=50000

extra_tag=30w_lr2e-3
#extra_tag="${extra_tag}_mixup_beta2"
#extra_tag="${extra_tag}_manifold-1,0,1,2,3"
#extra_tag="${extra_tag}_keepOrg"
#extra_tag="${extra_tag}_ratio2"
#extra_tag="${extra_tag}_speca"
#extra_tag="${extra_tag}_LNAll"
#extra_tag="${extra_tag}_consisCTC_consisMixup"

extra_parameter="--lr 2e-3"
#extra_parameter="--dropout 0.3"
#extra_parameter="--dropout 0.3"
# extra_parameter="--inter-mixup-beta 2"
# extra_parameter="${extra_parameter} --inter-mixup-keep-org True"
# extra_parameter="${extra_parameter} --inter-mixup-ratio 2"



extra_parameter=
#extra_parameter="${extra_parameter} "

exp_tag=

# CTC
#config_list=(purectc)

# Transformer
config_list=(base ctc)

# Conformer
#config_list=(base conformer ctc)

# PDS
#config_list=(purectc_pds_base_8)
#config_list=(pds_base_8)

# exp full name
exp_name=wrong10_30w_lr2e-3_delete_spm1000_speca

train_config=$(echo ${config_list[*]} | sed 's/ /,/g')

cmd="./run.sh
    --stage 1
    --stop_stage 1
    --gpu_num ${gpu_num}
    --update_freq ${update_freq}
    --train_config ${train_config}
    --max_tokens ${max_tokens}
    "

if [[ -n ${exp_name} ]]; then
    cmd="$cmd --exp_name ${exp_name}"
fi
if [[ -n ${exp_tag} ]]; then
    cmd="$cmd --exp_tag ${exp_tag}"
fi
if [[ -n ${extra_tag} ]]; then
    cmd="$cmd --extra_tag ${extra_tag}"
fi
if [[ -n ${extra_parameter} ]]; then
    cmd="$cmd --extra_parameter \"${extra_parameter}\""
fi

echo ${cmd}
eval ${cmd}
