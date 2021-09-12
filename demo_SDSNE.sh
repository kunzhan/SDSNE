# !/bin/bash

TRAINING_TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
TRAINING_LOG=${TRAINING_TIMESTAMP}.log
bash SDSNE_hyper_paras.sh | tee ./log/$TRAINING_LOG