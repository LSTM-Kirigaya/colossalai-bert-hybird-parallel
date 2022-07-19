import os
from common.processors import PROCESSORS

import torch
from torch.distributed import get_world_size
from transformers import BertConfig, BertTokenizer
from colossalai.logging import get_dist_logger
from colossalai.core import global_context as gpc
from colossalai.context import ParallelMode

from common.utils import CONFIG, ModelFromHF, get_model_size
from colossalai_utils.model_zoo.colo_bert import ColoBertMaskedLMLoss, ColoBertForMaskedLM, create_colo_bert_pipeline_model
from pretraining.pretrain_utils import get_dataloader_for_pretraining

_bert_base = dict(
    seq_length=512,
    vocab_size=50304,
    hidden_size=768,
    num_heads=12,
    depth=12,
    ff_size=3072,
    checkpoint=False,
    evaluation='ppl',
)

_bert_large = dict(
    seq_length=512,
    vocab_size=50304,
    hidden_size=1024,
    num_heads=16,
    depth=24,
    ff_size=3072,
    checkpoint=False,
    evaluation='ppl',
)

_bert_configurations = dict(bert=_bert_base, bert_base=_bert_base, bert_large=_bert_large)

_default_hyperparameters = dict(
    tokenize_mode='concat',
    batch_size=8,
    learning_rate=5e-5,
    weight_decay=1e-2,
    num_epochs=2,
    warmup_epochs=1,
    steps_per_epoch=100,
)


def build_data(args):
    # tokenizer = BertTokenizer.from_pretrained(args.vocab_file,
    #                                         do_lower_case=args.do_lower_case,
    #                                         max_len=512)
    # logger = get_dist_logger()
    # processor = PROCESSORS[args.task_name]()

    # train_data = get_train_dataloader(args, processor, logger)
    # test_data = get_eval_dataloader(args, tokenizer, processor, logger)
    dataloader = get_dataloader_for_pretraining(
        root=args.data,                         # ./pretrain_data/phase1/unbinned/parquet
        local_rank=gpc.get_local_rank(ParallelMode.DATA),
        vocab_file=args.vocab_file,             # bert-base-uncased
        global_batch_size=CONFIG['hyperparameter']['batch_size'],      # 32
    )

    if args.pretrained:
        return dataloader, None
    else:
        # TODO : train set and test set, but I still don't know GLUE_DATASET MRPC looks like
        ...


def build_model():
    model_cfg = CONFIG['model']
    bert_cfg = BertConfig(vocab_size=model_cfg['vocab_size'],
                          hidden_size=model_cfg['hidden_size'],
                          num_hidden_layers=model_cfg['depth'],
                          num_attention_heads=model_cfg['num_heads'],
                          intermediate_size=model_cfg['ff_size'],
                          max_position_embeddings=model_cfg['seq_length'],
                          use_cache=not CONFIG['model'].get('checkpoint', False))

    use_pipeline = 'parallel' in CONFIG and 'pipeline' in CONFIG['parallel'] and int(CONFIG['parallel']['pipeline']) > 1
    if use_pipeline:
        model = create_colo_bert_pipeline_model(bert_cfg)
    else:
        model = ModelFromHF(bert_cfg, ColoBertForMaskedLM)

    return model


def build_loss():
    return ColoBertMaskedLMLoss()


def build_optimizer(params):
    optimizer = torch.optim.AdamW(params,
                                  lr=CONFIG['hyperparameter']['learning_rate'],
                                  weight_decay=CONFIG['hyperparameter']['weight_decay'])
    return optimizer


def build_scheduler(epoch_steps, optimizer):
    from transformers.optimization import get_linear_schedule_with_warmup
    from colossalai.nn.lr_scheduler import LinearWarmupLR

    max_steps = epoch_steps * CONFIG['hyperparameter']['num_epochs']
    warmup_steps = epoch_steps * CONFIG['hyperparameter']['warmup_epochs']

    if CONFIG['method'] == 'colossalai':
        lr_scheduler = LinearWarmupLR(optimizer, total_steps=max_steps, warmup_steps=warmup_steps)
    else:
        lr_scheduler = get_linear_schedule_with_warmup(optimizer,
                                                       num_warmup_steps=warmup_steps,
                                                       num_training_steps=max_steps)

    return lr_scheduler


def bert_builder():
    model_type = CONFIG['model']['type']
    if model_type in _bert_configurations:
        for k, v in _bert_configurations[model_type].items():
            if k not in CONFIG['model']:
                CONFIG['model'][k] = v

    if 'hyperparameter' in CONFIG:
        for k, v in _default_hyperparameters.items():
            if k not in CONFIG['hyperparameter']:
                CONFIG['hyperparameter'][k] = v
    else:
        CONFIG['hyperparameter'] = _default_hyperparameters

    if 'numel' not in CONFIG['model']:
        CONFIG['model']['numel'] = get_model_size(build_model())

    return build_data, build_model, build_loss, build_optimizer, build_scheduler
