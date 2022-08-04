#!/bin/bash
NCCL_IB_HCA=`ibdev2netdev|awk '{print$1}'`
roce_PORT=":1"
NCCL_IB_HCA=${NCCL_IB_HCA}${roce_PORT}
NCCL_DEBUG=INFO
OMPI_MCA_btl_tcp_if_include=eth0
NCCL_SOCKET_IFNAME=eth0
NCCL_IB_DISABLE=1
export NCCL_IB_HCA
export NCCL_DEBUG
export OMPI_MCA_btl_tcp_if_include
export NCCL_SOCKET_IFNAME
export NCCL_IB_DISABLE

DIR=/home/notebook/data/personal/Megatron-LM
DATA_PATH=$DIR/my-bert_text_sentence
CHECKPOINT_PATH=$DIR/checkpoints/bert_345m
VOCAB_FILE=$DIR/bert-large-uncased-vocab.txt 

python pretrain_bert.py \
      --tensor-model-parallel-size 1 \
      --pipeline-model-parallel-size 1 \
      --num-layers 24 \
      --hidden-size 1024 \
      --num-attention-heads 16 \
      --micro-batch-size 16 \
      --global-batch-size 128 \
      --seq-length 512 \
      --max-position-embeddings 512 \
      --train-iters 1000000 \
      --data-path $DATA_PATH \
      --vocab-file $VOCAB_FILE \
      --data-impl mmap \
      --split 949,50,1 \
      --distributed-backend nccl \
      --lr 0.0001 \
      --lr-decay-style linear \
      --min-lr 1.0e-5 \
      --lr-decay-iters 990000 \
      --weight-decay 1e-2 \
      --clip-grad 1.0 \
      --lr-warmup-fraction .01 \
      --log-interval 100 \
      --save-interval 10000 \
      --eval-interval 1000 \
      --eval-iters 10 \
      --fp16