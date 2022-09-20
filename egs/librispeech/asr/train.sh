#! /bin/bash

# training the model

#sleep 2h

gpu_num=1
update_freq=1
max_tokens=50000

extra_tag=5w_lr2e-3
extra_tag="${extra_tag}_mixup_beta2.0"
#extra_tag="${extra_tag}_manifold5,6,7"
#extra_tag="${extra_tag}_noClipN"
extra_tag="${extra_tag}_keepOrg"
extra_tag="${extra_tag}_ratio2"
extra_tag="${extra_tag}_speca"


extra_parameter="--inter-mixup-beta 2"
#extra_parameter="${extra_parameter} --inter-mixup-layer ${layers}"
extra_parameter="${extra_parameter} --inter-mixup-keep-org True"
extra_parameter="${extra_parameter} --inter-mixup-ratio 2"

exp_tag=
#exp_tag=5wlr1e-3no2specaMixup2.0noConsis_v2
#exp_tag=5wlr1e-3no2specaMixup2.0noConsisDropOrigin

# Transformer
#config_list=(base)
#config_list=(pds_base_16)
#config_list=(pds_base_8)

# CTC
config_list=(base ctc mixup)
#config_list=(base ctc)
#config_list=(purectc_base)
#config_list=(purectc_pds_base_8)
#config_list=(purectc_pds_base_8_growth)
#config_list=(purectc_pds_base_8_growth_fusion256)
#config_list=(purectc_pds_base_16)
#config_list=(purectc_pds_base_16_growth)
#config_list=(purectc_pds_base_16_growth_fusion256)
#config_list=(purectc_pds_base_16_growth_fusion320)

# conformer
#config_list=(base ctc conformer)
#config_list=(big conformer)
#config_list=(pds_base_4 conformer)
#config_list=(pds_base_16 conformer)
#config_list=(pds_base_32 conformer)
#config_list=(pds_big_8 conformer)
#config_list=(pds_big_16 conformer)
#config_list=(pds_big_32 conformer)
#config_list=(pds_base_8_growth_fusion256 conformer)

# growth validation
#config_list=(pds_base_8_growth)
#config_list=(pds_base_8_growth_fusion256)
#config_list=(pds_base_16_growth_fusion256)
#config_list=(pds_base_16_growth)

# compare with Effective
#config_list=(purectc_base_compare)
#config_list=(purectc_pds_base_8_compare)
#config_list=(purectc_pds_base_8_compare2)
#config_list=(EffecientConformerCTCSmall)
#config_list=(purectc_pds_base_16)

# exp full name
exp_name=

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
