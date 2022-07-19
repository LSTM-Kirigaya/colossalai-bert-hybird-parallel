
BERT_CONFIG_PATH='./colossalai_utils/bert_config_tp2p5d.json'
PY_FILE_PATH='./pretraining/run_pretraining.py'
DATA_PATH='./pretrain_data/phase1/unbinned/parquet'
VOCAB_FILE='bert-base-uncased'

export PYTHONPATH=$PWD

torchrun --nproc_per_node 8 \
    $PY_FILE_PATH \
    --config $BERT_CONFIG_PATH \
    --vocab_file $VOCAB_FILE \
    --data $DATA_PATH