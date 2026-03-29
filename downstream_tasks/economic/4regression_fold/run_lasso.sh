#!/bin/bash
# filepath: /home/liyong/code/svpretrain/economic/4regression_fold/run_lasso.sh

FEATURE_PATH="/home/liyong/code/svpretrain/urban2vec_/urban2vec_step1/embeddingny_chi1_vit.csv"
LABEL_PATH="/home/huangyj/representation/data/inter/US/LosAngeles/labels_norm.csv"
# 修改：将Python列表格式改为shell脚本中的逗号分隔字符串
TARGETS="logcrime,logpetty,walkbike_per_cbg,publictrans_per_cbg,drove_alone_per_cbg,estvmiles,estpmiles,estvtrp,estptrp,obesitycru,diabetescr,lpacrudepr,mhlthcrude,phlthcrude,cancercrud,mhincome_cbg,povertyline_below100,povertyline_below200"
OUTPUT_DIR="./output"
FEATURE_LEN=50

cd /home/liyong/code/svpretrain/economic/4regression_fold
python lasso1.py \
    --feature_path "$FEATURE_PATH" \
    --label_path "$LABEL_PATH" \
    --targets "$TARGETS" \
    --output_dir "$OUTPUT_DIR" \
    --feature_len $FEATURE_LEN