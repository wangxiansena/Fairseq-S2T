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
time=$(date "+%m%d")

stage=0
stop_stage=0

######## hardware ########
# devices
device=()
gpu_num=8
update_freq=1

root_dir=~/st
code_dir=${root_dir}/Fairseq-S2T
pwd_dir=$PWD

# dataset
src_lang=en
tgt_lang=de
lang=${src_lang}-${tgt_lang}

dataset=mustc
task=translation
src_vocab_type=unigram
tgt_vocab_type=unigram
src_vocab_size=10000
tgt_vocab_size=10000
share_dict=1
lcrm=0
tokenizer=1

use_specific_dict=1
specific_prefix=st
specific_dir=${root_dir}/data/${dataset}/st_tok
src_vocab_prefix=spm_unigram10000_st_share
tgt_vocab_prefix=spm_unigram10000_st_share

org_data_dir=${root_dir}/data/${dataset}
data_dir=${root_dir}/data/${dataset}/mt
train_subset=train
valid_subset=dev
trans_subset=tst-COMMON
test_subset=test

# exp
exp_prefix=${time}
extra_tag=
extra_parameter=
exp_tag=baseline
exp_name=

# config
train_config=base_s

# training setting
fp16=1
max_tokens=4096
step_valid=0
bleu_valid=0

# decoding setting
sacrebleu=1
dec_model=checkpoint_best.pt
n_average=10
beam_size=5
len_penalty=1.0

if [[ ${use_specific_dict} -eq 1 ]]; then
    exp_prefix=${exp_prefix}_${specific_prefix}
    data_dir=${data_dir}/${specific_prefix}
else
    if [[ "${tgt_vocab_type}" == "char" ]]; then
        vocab_name=char
        exp_prefix=${exp_prefix}_char
    else
        if [[ ${src_vocab_size} -ne ${tgt_vocab_size} || "${src_vocab_type}" -ne "${tgt_vocab_type}" ]]; then
            src_vocab_name=${src_vocab_type}${src_vocab_size}
            tgt_vocab_name=${tgt_vocab_type}${tgt_vocab_size}
            vocab_name=${src_vocab_name}_${tgt_vocab_name}
        else
            vocab_name=${tgt_vocab_type}${tgt_vocab_size}
            src_vocab_name=${vocab_name}
            tgt_vocab_name=${vocab_name}
        fi
    fi
    data_dir=${data_dir}/${vocab_name}
    src_vocab_prefix=spm_${src_vocab_name}_${src_lang}
    tgt_vocab_prefix=spm_${tgt_vocab_name}_${tgt_lang}
    if [[ $share_dict -eq 1 ]]; then
        data_dir=${data_dir}_share
        src_vocab_prefix=spm_${vocab_name}_share
        tgt_vocab_prefix=spm_${vocab_name}_share
    fi
fi
if [[ ${lcrm} -eq 1 ]]; then
    data_dir=${data_dir}_lcrm
    exp_prefix=${exp_prefix}_lcrm
fi
if [[ ${tokenizer} -eq 1 ]]; then
    data_dir=${data_dir}_tok
    exp_prefix=${exp_prefix}_tok
fi

. ./local/parse_options.sh || exit 1;

# full path
if [[ -z ${exp_name} ]]; then
    config_string=${train_config//,/_}
    exp_name=${exp_prefix}_${config_string}_${exp_tag}
    if [[ -n ${extra_tag} ]]; then
        exp_name=${exp_name}_${extra_tag}
    fi
fi
model_dir=${root_dir}/checkpoints/${dataset}/mt/${exp_name}

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data Download"
    # pass
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    echo "stage 0: MT Data Preparation"
    if [[ ! -e ${data_dir} ]]; then
        mkdir -p ${data_dir}
    fi
    if [[ ! -e ${data_dir}/data ]]; then
        mkdir -p ${data_dir}/data
    fi

    if [[ ! -f ${data_dir}/${src_vocab_prefix}.txt || ! -f ${data_dir}/${tgt_vocab_prefix}.txt ]]; then
        if [[ ${use_specific_dict} -eq 0 ]]; then
            cmd="python ${code_dir}/examples/speech_to_text/prep_mt_data.py
                --data-root ${org_data_dir}
                --output-root ${data_dir}
                --splits ${train_subset},${valid_subset},${trans_subset}
                --src-lang ${src_lang}
                --tgt-lang ${tgt_lang}
                --src-vocab-type ${src_vocab_type}
                --tgt-vocab-type ${tgt_vocab_type}
                --src-vocab-size ${src_vocab_size}
                --tgt-vocab-size ${tgt_vocab_size}"
        else
            cp -r ${specific_dir}/${src_vocab_prefix}.* ${data_dir}
            cp ${specific_dir}/${tgt_vocab_prefix}.* ${data_dir}

            cmd="python ${code_dir}/examples/speech_to_text/prep_mt_data.py
                --data-root ${org_data_dir}
                --output-root ${data_dir}
                --splits ${train_subset},${valid_subset},${trans_subset}
                --src-lang ${src_lang}
                --tgt-lang ${tgt_lang}
                --src-vocab-prefix ${src_vocab_prefix}
                --tgt-vocab-prefix ${tgt_vocab_prefix}"
        fi
        if [[ $share_dict -eq 1 ]]; then
            cmd="$cmd
                --share"
        fi
        if [[ $tokenizer -eq 1 ]]; then
            cmd="$cmd
                --tokenizer"
        fi
        if [[ ${lcrm} -eq 1 ]]; then
            cmd="$cmd
                --lowercase-src
                --rm-punc-src"
        fi
        echo -e "\033[34mRun command: \n${cmd} \033[0m"
        [[ $eval -eq 1 ]] && eval ${cmd}
    fi

    cmd="python ${code_dir}/fairseq_cli/preprocess.py
        --source-lang ${src_lang} --target-lang ${tgt_lang}
        --trainpref ${data_dir}/data/${train_subset}
        --validpref ${data_dir}/data/${valid_subset}
        --testpref ${data_dir}/data/${trans_subset}
        --destdir ${data_dir}/data-bin
        --srcdict ${data_dir}/${src_vocab_prefix}.txt
        --tgtdict ${data_dir}/${tgt_vocab_prefix}.txt
        --workers 64"

    echo -e "\033[34mRun command: \n${cmd} \033[0m"
    [[ $eval -eq 1 ]] && eval ${cmd}
fi

data_dir=${data_dir}/data-bin

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: MT Network Training"
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
        --source-lang ${src_lang}
        --target-lang ${tgt_lang}
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
        no_epoch_checkpoints=0
        save_interval_updates=500
        keep_interval_updates=10
    fi
    if [[ $bleu_valid -eq 1 ]]; then
        cmd="$cmd
        --eval-bleu
        --eval-bleu-args '{\"beam\": 1}'
        --eval-tokenized-bleu
        --eval-bleu-remove-bpe
        --best-checkpoint-metric bleu
        --maximize-best-checkpoint-metric"
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

    log=${model_dir}/train.log
    cmd="nohup ${cmd} >> ${log} 2>&1 &"
    if [[ $eval -eq 1 ]]; then
		eval $cmd
		sleep 2s
		tail -n "$(wc -l ${log} | awk '{print $1+1}')" -f ${log}
	fi
fi
wait

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: MT Decoding"
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

    suffix=beam${beam_size}_alpha${len_penalty}_tokens${max_tokens}
    if [[ ${n_average} -ne 1 ]]; then
        suffix=${suffix}_${n_average}
    fi
	result_file=${model_dir}/decode_result_${suffix}
	[[ -f ${result_file} ]] && rm ${result_file}

    test_subset=${test_subset//,/ }
	for subset in ${test_subset[@]}; do
  		cmd="python ${code_dir}/fairseq_cli/generate.py
        ${data_dir}
        --source-lang ${src_lang}
        --target-lang ${tgt_lang}
        --gen-subset ${subset}
        --task ${task}
        --path ${model_dir}/${dec_model}
        --results-path ${model_dir}
        --max-tokens ${max_tokens}
        --beam ${beam_size}
        --lenpen ${len_penalty}
        --post-process sentencepiece"

        if [[ ${sacrebleu} -eq 1 ]]; then
            cmd="${cmd}
        --scoring sacrebleu"
            if [[ ${tokenizer} -eq 1 ]]; then
                cmd="${cmd}
        --tokenizer moses
        --source-lang ${src_lang}
        --target-lang ${tgt_lang}"
            fi
        fi

    	echo -e "\033[34mRun command: \n${cmd} \033[0m"

        if [[ $eval -eq 1 ]]; then
    	    eval $cmd
    	    tail -n 1 ${model_dir}/generate-${subset}.txt >> ${result_file}
            mv ${model_dir}/generate-${subset}.txt ${model_dir}/generate-${subset}-${suffix}.txt
            mv ${model_dir}/translation-${subset}.txt ${model_dir}/translation-${subset}-${suffix}.txt
        fi
	done
    cat ${result_file}
fi
