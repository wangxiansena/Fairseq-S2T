#! /bin/bash

# training the model

#sleep 2h

gpu_num=1
update_freq=1
max_tokens=50000

extra_tag=5w_lr1e-3
#extra_tag="${extra_tag}_mixup_beta0.5"
#extra_tag="${extra_tag}_manifold-1,0,1,2,3"
#extra_tag="${extra_tag}_noClipN"
#extra_tag="${extra_tag}_keepOrg"
#extra_tag="${extra_tag}_ratio2"
extra_tag="${extra_tag}_speca"
#extra_tag="${extra_tag}_posEmbedAfter"
#extra_tag="${extra_tag}_encoderNormBeforeF"
#extra_tag="${extra_tag}_BN"
#extra_tag="${extra_tag}_LN"
#extra_tag="${extra_tag}_LNAll"
#extra_tag="${extra_tag}_EENT,ENSET"
#extra_tag="${extra_tag}_consisCTC_consisMixup"
extra_tag="${extra_tag}_twostage"

#extra_parameter="--inter-mixup-beta 0.5"
#extra_parameter="${extra_parameter} --inter-mixup-keep-org True"
#extra_parameter="${extra_parameter} --inter-mixup-ratio 2"
extra_parameter="${extra_parameter} --max-epoch 200"

exp_tag=
#exp_tag=5wlr1e-3no2specaMixup2.0noConsis_v2
#exp_tag=5wlr1e-3no2specaMixup2.0noConsisDropOrigin

# Transformer
#config_list=(base)
#config_list=(pds_base_16)
#config_list=(pds_base_8)

# CTC
#config_list=(base ctc conformer mixup)
config_list=(base ctc)
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
#exp_name=0924_base_ctc_mixup_baseline_5w_lr1e-3_mixup_beta2_manifold-1,0,1,2,3_TN_EENT,ENSET
exp_name=1019_base_ctc_mixup_baseline_5w_lr1e-3_mixup_beta0.5_keepOrg_ratio2_speca_twostage

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
