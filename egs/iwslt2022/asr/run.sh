#! /bin/bash

# Processing MuST-C Datasets

# Copyright 2021 Natural Language Processing Laboratory 
# Xu Chen (xuchenneu@163.com)

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
#set -u
set -o pipefail
export PYTHONIOENCODING=UTF-8

eval=1
time=$(date "+%m%d_%H%M")

stage=0
stop_stage=0

######## hardware ########
# devices
#device=()
gpu_num=8
update_freq=1

root_dir=~/st
code_dir=${root_dir}/Fairseq-S2T
pwd_dir=$PWD

# dataset
src_lang=en
tgt_lang=zh
lang=${src_lang}-${tgt_lang}

dataset=iwslt2022
task=speech_to_text
vocab_type=unigram
vocab_size=5000
speed_perturb=0
lcrm=1
tokenizer=0
use_raw_audio=0

use_specific_dict=1
specific_prefix=unified
specific_dir=${root_dir}/data/iwslt2022/vocab
asr_vocab_prefix=spm_en

org_data_dir=${root_dir}/data/${dataset}
data_dir=${root_dir}/data/${dataset}/asr
train_split=train_covost,train_eu,train_iwslt,train_mustc_ende,train_voxpopuil,train_mustc_enzh,train_ted
valid_split=dev
test_split=tst-COMMON
test_subset=tst-COMMON

# exp
exp_prefix=$(date "+%m%d")
extra_tag=
extra_parameter=
exp_tag=baseline
exp_name=

# config
train_config=base
data_config=config.yaml

# training setting
fp16=1
max_tokens=40000
step_valid=0

# decoding setting
cer=0
dec_model=checkpoint_best.pt
n_average=10
beam_size=5
len_penalty=1.0

if [[ ${speed_perturb} -eq 1 ]]; then
    data_dir=${data_dir}_sp
    exp_prefix=${exp_prefix}_sp
fi
if [[ ${lcrm} -eq 1 ]]; then
    data_dir=${data_dir}_lcrm
    exp_prefix=${exp_prefix}_lcrm
fi
if [[ ${use_specific_dict} -eq 1 ]]; then
    data_dir=${data_dir}_${specific_prefix}
    exp_prefix=${exp_prefix}_${specific_prefix}
fi
if [[ ${tokenizer} -eq 1 ]]; then
    data_dir=${data_dir}_tok
    exp_prefix=${exp_prefix}_tok
fi
if [[ ${use_raw_audio} -eq 1 ]]; then
    data_dir=${data_dir}_raw
    exp_prefix=${exp_prefix}_raw
fi

. ./local/parse_options.sh || exit 1;

if [[ -z ${exp_name} ]]; then
    config_string=${train_config//,/_}
    exp_name=${exp_prefix}_${config_string}_${exp_tag}
    if [[ -n ${extra_tag} ]]; then
        exp_name=${exp_name}_${extra_tag}
    fi
fi
model_dir=${root_dir}/checkpoints/${dataset}/asr/${exp_name}

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data Download"
    # pass
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: ASR Data Preparation"
    if [[ ! -e ${data_dir} ]]; then
        mkdir -p ${data_dir}
    fi
    feature_zip=fbank80.zip
    if [[ ${speed_perturb} -eq 1 ]]; then
        feature_zip=fbank80_sp.zip
    fi
    if [[ ! -f ${data_dir}/${feature_zip} && -f ${data_dir}/../${feature_zip} ]]; then
        ln -s ${data_dir}/../${feature_zip} ${data_dir}
    fi

    cmd="python ${code_dir}/examples/speech_to_text/prep_audio_data.py
        --data-root ${org_data_dir}
        --output-root ${data_dir}
        --task asr
        --src-lang ${src_lang}
        --splits ${valid_split},${test_split},${train_split}
        --vocab-type ${vocab_type}
        --vocab-size ${vocab_size}"

    if [[ ${use_raw_audio} -eq 1 ]]; then
        cmd="$cmd
        --raw"
    fi
    if [[ ${use_specific_dict} -eq 1 ]]; then
        cp -r ${specific_dir}/${asr_vocab_prefix}.* ${data_dir}
        cmd="$cmd
        --asr-prefix ${asr_vocab_prefix}"
    fi
    if [[ ${speed_perturb} -eq 1 ]]; then
        cmd="$cmd
        --speed-perturb"
    fi
    if [[ ${lcrm} -eq 1 ]]; then
        cmd="$cmd
        --lowercase-src
        --rm-punc-src"
    fi
    if [[ ${tokenizer} -eq 1 ]]; then
        cmd="$cmd
        --tokenizer"
    fi

    echo -e "\033[34mRun command: \n${cmd} \033[0m"
    [[ $eval -eq 1 ]] && eval ${cmd}

    if [[ -f ${data_dir}/${feature_zip} && ! -f ${data_dir}/../${feature_zip} ]]; then
        mv ${data_dir}/${feature_zip} ${data_dir}/..
        ln -s ${data_dir}/../${feature_zip} ${data_dir}
    fi
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: ASR Network Training"
    [[ ! -d ${data_dir} ]] && echo "The data dir ${data_dir} is not existing!" && exit 1;

    if [[ -z ${device} || ${#device[@]} -eq 0 ]]; then
		if [[ ${gpu_num} -eq 0 ]]; then
			device=""
		else
        	source ./local/utils.sh
        	device=$(get_devices $gpu_num 0)
		fi
    fi

    echo -e "dev=${device} data=${data_dir} model=${model_dir}"

    if [[ ! -d ${model_dir} ]]; then
        mkdir -p ${model_dir}
    else
        echo "${model_dir} exists."
    fi

    cp ${BASH_SOURCE[0]} ${model_dir}
    cp ${PWD}/train.sh ${model_dir}

    extra_parameter="${extra_parameter}
        --train-config ${pwd_dir}/conf/basis.yaml"
    cp ${pwd_dir}/conf/basis.yaml ${model_dir}
    config_list="${train_config//,/ }"
    idx=1
    for config in ${config_list[@]}
    do
        config_path=${pwd_dir}/conf/${config}.yaml
        if [[ ! -f ${config_path} ]]; then
            echo "No config file ${config_path}"
            exit
        fi
        cp ${config_path} ${model_dir}

        extra_parameter="${extra_parameter}
        --train-config${idx} ${config_path}"
        idx=$((idx + 1))
    done

    cmd="python3 -u ${code_dir}/fairseq_cli/train.py
        ${data_dir}
        --config-yaml ${data_config}
        --task ${task}
        --max-tokens ${max_tokens}
        --skip-invalid-size-inputs-valid-test
        --update-freq ${update_freq}
        --log-interval 100
        --save-dir ${model_dir}
        --tensorboard-logdir ${model_dir}"

	if [[ -n ${extra_parameter} ]]; then
        cmd="${cmd}
        ${extra_parameter}"
    fi
	if [[ ${gpu_num} -gt 0 ]]; then
		cmd="${cmd}
        --distributed-world-size $gpu_num
        --ddp-backend no_c10d"
	fi
    if [[ $fp16 -eq 1 ]]; then
        cmd="${cmd}
        --fp16"
    fi
    if [[ $step_valid -eq 1 ]]; then
        validate_interval=1
        save_interval=1
        keep_last_epochs=10
        no_epoch_checkpoints=0
        save_interval_updates=500
        keep_interval_updates=10
    else
        validate_interval=1
        keep_last_epochs=10
    fi
    if [[ -n $no_epoch_checkpoints && $no_epoch_checkpoints -eq 1 ]]; then
        cmd="$cmd
        --no-epoch-checkpoints"
    fi
    if [[ -n $validate_interval ]]; then
        cmd="${cmd}
        --validate-interval $validate_interval "
    fi
    if [[ -n $save_interval ]]; then
        cmd="${cmd}
        --save-interval $save_interval "
    fi
    if [[ -n $keep_last_epochs ]]; then
        cmd="${cmd}
        --keep-last-epochs $keep_last_epochs "
    fi
    if [[ -n $save_interval_updates ]]; then
        cmd="${cmd}
        --save-interval-updates $save_interval_updates"
        if [[ -n $keep_interval_updates ]]; then
        cmd="${cmd}
        --keep-interval-updates $keep_interval_updates"
        fi
    fi

    echo -e "\033[34mRun command: \n${cmd} \033[0m"

    # save info
    log=./history.log
    echo "${time} | ${device} | ${data_dir} | ${exp_name} | ${model_dir} " >> $log
    tail -n 50 ${log} > tmp.log
    mv tmp.log $log
    export CUDA_VISIBLE_DEVICES=${device}

    cmd="nohup ${cmd} >> ${model_dir}/train.log 2>&1 &"
    if [[ $eval -eq 1 ]]; then
		eval $cmd
		sleep 2s
		tail -n "$(wc -l ${model_dir}/train.log | awk '{print $1+1}')" -f ${model_dir}/train.log
	fi
fi
wait

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: ASR Decoding"
    if [[ ${n_average} -ne 1 ]]; then
        # Average models
		dec_model=avg_${n_average}_checkpoint.pt

        if [[ ! -f ${model_dir}/${dec_model} ]]; then
            cmd="python ${code_dir}/scripts/average_checkpoints.py
            --inputs ${model_dir}
            --num-best-checkpoints ${n_average}
            --output ${model_dir}/${dec_model}"
            echo -e "\033[34mRun command: \n${cmd} \033[0m"
            [[ $eval -eq 1 ]] && eval $cmd
        fi
	else
		dec_model=${dec_model}
	fi

    if [[ -z ${device} || ${#device[@]} -eq 0 ]]; then
		if [[ ${gpu_num} -eq 0 ]]; then
			device=""
		else
        	source ./local/utils.sh
        	device=$(get_devices $gpu_num 0)
		fi
    fi
    export CUDA_VISIBLE_DEVICES=${device}

	result_file=${model_dir}/decode_result
	[[ -f ${result_file} ]] && rm ${result_file}

    test_subset=${test_subset//,/ }
	for subset in ${test_subset[@]}; do
        subset=${subset}
  		cmd="python ${code_dir}/fairseq_cli/generate.py
        ${data_dir}
        --config-yaml ${data_config}
        --gen-subset ${subset}
        --task speech_to_text
        --path ${model_dir}/${dec_model}
        --results-path ${model_dir}
        --max-tokens ${max_tokens}
        --beam ${beam_size}
        --lenpen ${len_penalty}
        --scoring wer
        --wer-tokenizer 13a
        --wer-lowercase
        --wer-remove-punct
        "

        if [[ ${cer} -eq 1 ]]; then
            cmd="${cmd}
        --wer-char-level"
        fi

    	echo -e "\033[34mRun command: \n${cmd} \033[0m"

        if [[ $eval -eq 1 ]]; then
    	    eval $cmd
    	    tail -n 1 ${model_dir}/generate-${subset}.txt >> ${result_file}
        fi
	done

    cat ${result_file}
fi
