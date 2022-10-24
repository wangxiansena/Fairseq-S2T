from numpy import source
source_file = "/home/wangjie/st/checkpoints/TEDLIUM_release-3/asr/wrong9_15w_lr2e-3_delete_spm2000_speca/translation-test-beam5_alpha1.0_tokens50000_wer_10.txt"

with open(source_file, 'r', encoding = 'utf-8') as sourceFile, open("gold.txt", 'w', encoding='utf-8') as goldFile, open("pred.txt", 'w', encoding='utf-8') as predFile:
    lines = sourceFile.readlines()
    for line in lines:
        line = line.split('\t')
        goldFile.write(line[2] + '\n')
        predFile.write(line[3] + '\n')
