import os

from colossalai.logging import disable_existing_loggers
from torch.distributed import get_rank, get_world_size

from common.helper import bert_builder
from common.utils import CONFIG, load_config, print_log, AsyncMemoryMonitor
from common.train import _train
from colossalai_utils.utils import init_w_col
from torch_utils.utils import init_w_torch

_method = {
    'torch': init_w_torch,
    'colossalai': init_w_col,
}

_builder = {
    'bert': bert_builder,
}


def run_bert(args):
    method = CONFIG['method']
    model = CONFIG['model']['type']
    model_type = model.split('_')[0]

    model, train_data, test_data, criterion, optimizer, scaler, lr_scheduler = _method[method](_builder[model_type], args)

    use_pipeline = 'parallel' in CONFIG and 'pipeline' in CONFIG['parallel'] and int(CONFIG['parallel']['pipeline']) > 1

    rank = get_rank()
    world_size = get_world_size()

    mem_monitor = None
    if CONFIG.get('use_mem_monitor'):
        mem_monitor = AsyncMemoryMonitor(rank)

    numel = CONFIG['model']['numel']
    if numel < 1e9:
        msg = f'{numel / 1e6:.3f} M'
    else:
        msg = f'{numel / 1e9:.3f} B'
    print_log(f'Model is built (parameter size = {msg}).')

    print_log('Benchmark start.')

    if use_pipeline:
        import colossalai.nn as col_nn
        from colossalai.engine.schedule import PipelineSchedule
        from colossalai.utils import MultiTimer, get_dataloader
        from colossalai.logging import get_dist_logger
        from colossalai.trainer import Trainer, hooks

        def batch_data_process_func(batch_data):
            data = {
                'input_ids': batch_data['input_ids'],
                'token_type_ids': batch_data['token_type_ids'],
                'attention_mask': batch_data['attention_mask']
            }
            labels = batch_data['labels']
            return data, labels

        timer = MultiTimer()
        schedule = PipelineSchedule(num_microbatches=2, batch_data_process_func=batch_data_process_func)
        engine = model
        logger = get_dist_logger()
        trainer = Trainer(engine=engine, timer=timer, logger=logger, schedule=schedule)

        hook_list = [
            hooks.LossHook(),
            hooks.AccuracyHook(col_nn.metric.Accuracy()),
            hooks.LogMetricByEpochHook(logger),
            hooks.LRSchedulerHook(lr_scheduler, by_epoch=True)
        ]

        trainer.fit(train_dataloader=train_data,
                    epochs=CONFIG['hyperparameter']['num_epochs'],
                    test_dataloader=test_data,
                    test_interval=1,
                    hooks=hook_list,
                    display_progress=True)

    else:
        for epoch in range(CONFIG['hyperparameter']['num_epochs']):
            _train(epoch, rank, world_size, train_data, model, criterion, optimizer, lr_scheduler, scaler, mem_monitor)

    print_log('Benchmark complete.')


if __name__ == '__main__':
    disable_existing_loggers()
    args = load_config()
    args.pretrained = True

    CONFIG['log_path'] = os.environ.get('LOG', '.')
    os.makedirs(CONFIG['log_path'], exist_ok=True)
    print_log(f'Initializing {CONFIG["method"]} ...')

    run_bert(args)
