set -e

ref=$1
gen=$2
tokenizer=$3
lang=$4
lang_pair=en-${lang}

record=$(mktemp -t temp.record.XXXXXX)
if [[ ${tokenizer} -eq 1 ]]; then
    echo "MultiBLEU" > ${record}
    cmd="multi-bleu.perl ${ref} < ${gen}"
#    echo $cmd
    eval $cmd | head -n 1 >> ${record}

    cmd="detokenizer.perl -l ${lang} --threads 32 < ${ref} > ${ref}.detok"
#    echo $cmd
#    echo
    eval $cmd
    cmd="detokenizer.perl -l ${lang} --threads 32 < ${gen} > ${gen}.detok"
#    echo $cmd
#    echo
    eval $cmd
    ref=${ref}.detok
    gen=${gen}.detok
fi

echo "SacreBLEU" > ${record}
cmd="cat ${gen} | sacrebleu ${ref} -m bleu -w 4 -l ${lang_pair}"
#echo $cmd
eval $cmd >> ${record}
cat ${record}
rm ${record}