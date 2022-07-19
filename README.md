# Bert

![Still In Progress](https://img.shields.io/badge/-Still%20In%20Progress-orange)

- [x] tp
- [ ] pp
- [ ] zero

Bert Benchmark with data parallel, tensor parallel(tp), pipeline parallel(pp) and ZeRO.

## Setup
1. Install dependencies if you do not have them
```
pip install -r requirement.txt
```

2. Add root dir into PYTHONPATH
```
export PYTHONPATH=$(dirname "$PWD"):$PYTHONPATH
```

## Bert Usage

1. Prepare datasets and tokenizers from HuggingFace Hub if necessary (e.g. we provide an example of training `wikitext-2`).

2. Run benchmark with one of the systems to evaluate
```
DATA=/PATH/TO/DATASET TOKENIZER=/PATH/TO/TOKENIZER LOG=/PATH/TO/LOG torchrun --nproc_per_node=NUM_GPUS run.py --config=CONFIG_FILE
```

Here, <CONFIG_FILE> is a json. A possible command is as follow:

```bash
DATA=huggingface/datasets/wikitext/wikitext-2 TOKENIZER=huggingface/tokenizers/bert/bert-base-cased LOG=log torchrun --nproc_per_node=4 run.py --config=colossalai_utils/bert_config_zero.json
```