import os

from colossalai.logging import disable_existing_loggers

from common.helper import bert_builder
from colossalai_utils.utils import init_w_col
from common.train import train
from common.utils import CONFIG, load_config, print_log


def run_bert():
    train(*init_w_col(bert_builder))


if __name__ == '__main__':
    disable_existing_loggers()
    args = load_config()

    CONFIG['log_path'] = os.environ.get('LOG', '.')
    os.makedirs(CONFIG['log_path'], exist_ok=True)
    print_log(f'Initializing {CONFIG["method"]} ...')

    run_bert()
