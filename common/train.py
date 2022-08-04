import math
import time

import torch
from torch.distributed import all_reduce, get_rank, get_world_size
from tqdm import tqdm

from common.utils import CONFIG, AsyncMemoryMonitor, print_log, get_tflops


def _train(epoch, rank, world_size, train_dataloader, model, criterion, optimizer, lr_scheduler, scaler):
    use_autocast = 'fp16' in CONFIG and CONFIG['fp16'].get('enabled', True)
    clip_grad_norm = CONFIG.get('gradient_clipping', 0.)

    model.train()

    num_steps = len(train_dataloader)
    if 'steps_per_epoch' in CONFIG['hyperparameter'] and CONFIG['hyperparameter']['steps_per_epoch'] < num_steps:
        num_steps = CONFIG['hyperparameter']['steps_per_epoch']
    progress = train_dataloader

    if rank == 0:
        progress = tqdm(progress, desc=f"[Epoch {epoch} / Train]")

    train_loss = torch.zeros(()).to(torch.float).to(rank)
    used_time = 0.
    num_steps = 0
    num_samples = torch.zeros(()).to(torch.int).to(rank)
    num_tokens = torch.zeros(()).to(torch.int).to(rank)

    for batch in progress:
        fwd_start = time.time()
        optimizer.zero_grad()

        labels = batch.pop('labels')
        batch_size = None
        batch_tokens = None
        if isinstance(labels, torch.Tensor):
            labels = labels.to(rank)
            batch_size = labels.size(0)
            batch_tokens = labels.numel()
        else:
            for k, v in labels.items():
                labels[k] = v.to(rank)
                if batch_size is None:
                    batch_size = v.size(0)
                if batch_tokens is None:
                    batch_tokens = v.numel()

        for k, v in batch.items():
            batch[k] = v.to(rank)

        if use_autocast:
            with torch.cuda.amp.autocast():
                outputs = model(**batch)
        else:
            outputs = model(**batch)

        loss = criterion(outputs, labels)
        train_loss += loss

        fwd_end = time.time()

        bwd_start = time.time()

        optimizer.backward(loss)
        optimizer.step()
        lr_scheduler.step()

        bwd_end = time.time()

        num_steps += 1
        num_samples += batch_size
        num_tokens += batch_tokens

        fwd_time = fwd_end - fwd_start
        bwd_time = bwd_end - bwd_start
        batch_time = fwd_time + bwd_time
        used_time += batch_time

        if rank == 0:
            progress.set_postfix(loss=loss.item(),
                                 lr=lr_scheduler.get_last_lr()[0],
                                 time_forward=fwd_time,
                                 time_backward=bwd_time,
                                 throughput=batch_size * world_size / (batch_time + 1e-12),
                                 tflops=get_tflops(batch_time, batch_tokens * world_size))

    all_reduce(train_loss)
    all_reduce(num_samples)
    all_reduce(num_tokens)

    msg = f'[Epoch {epoch} / Train]: Loss = {train_loss.item() / (world_size * num_steps):.3f}'
    msg += f' | Throughput = {num_samples.item() / (used_time + 1e-12):.3f} samples/sec'
    msg += f' | TFLOPS = {get_tflops(used_time, num_tokens.item()):.3f}'
    print_log(msg)


def _test(epoch, rank, world_size, test_dataloader, model, criterion, mem_monitor):
    use_autocast = 'fp16' in CONFIG and CONFIG['fp16'].get('enabled', True)
    evaluation = CONFIG['model']['evaluation']

    model.eval()

    num_steps = len(test_dataloader)
    if 'steps_per_epoch' in CONFIG['hyperparameter'] and CONFIG['hyperparameter']['steps_per_epoch'] < num_steps:
        num_steps = CONFIG['hyperparameter']['steps_per_epoch']
    progress = range(num_steps)
    if rank == 0:
        progress = tqdm(progress, desc=f"[Epoch {epoch} / Test]")

    test_loss = torch.zeros(()).to(torch.float).to(rank)
    used_time = 0.
    num_steps = 0
    num_samples = torch.zeros(()).to(torch.int).to(rank)
    num_tokens = torch.zeros(()).to(torch.int).to(rank)
    correct = torch.zeros(()).to(torch.int).to(rank)

    data_iter = iter(test_dataloader)

    if mem_monitor is not None:
        mem_monitor.start()

    with torch.no_grad():
        for _ in progress:
            batch_start = time.time()

            batch = next(data_iter)

            labels = batch.pop('labels')
            batch_size = None
            batch_tokens = None
            if isinstance(labels, torch.Tensor):
                labels = labels.to(rank)
                batch_size = labels.size(0)
                batch_tokens = labels.numel()
            else:
                for k, v in labels.items():
                    labels[k] = v.to(rank)
                    if batch_size is None:
                        batch_size = v.size(0)
                    if batch_tokens is None:
                        batch_tokens = v.numel()

            for k, v in batch.items():
                batch[k] = v.to(rank)
            if use_autocast:
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
            else:
                outputs = model(**batch)

            loss = criterion(outputs, labels)
            test_loss += loss

            batch_end = time.time()

            num_steps += 1
            num_samples += batch_size
            num_tokens += batch_tokens

            batch_time = batch_end - batch_start
            used_time += batch_time

            if rank == 0:
                metrics = dict(loss=loss.item(),
                               step_time=batch_time,
                               throughput=batch_size * world_size / (batch_time + 1e-12),
                               ppl=math.exp(loss.item(),
                               tflops=get_tflops(batch_time, batch_tokens * world_size))
                progress.set_postfix(**metrics)

    all_reduce(test_loss)
    reduced_loss = test_loss.item() / (world_size * num_steps)
    all_reduce(num_samples)
    all_reduce(num_tokens)

    msg = f'[Epoch {epoch} / Test]: Loss = {reduced_loss:.3f}'
    msg += f' | Perplexity = {math.exp(reduced_loss):.3f}'
    msg += f' | Throughput = {num_samples.item() / (used_time + 1e-12):.3f} samples/sec'
    msg += f' | TFLOPS = {get_tflops(used_time, num_tokens.item()):.3f}'
    print_log(msg)


def train(model, train_data, test_data, criterion, optimizer, scaler, lr_scheduler):
    rank = get_rank()
    world_size = get_world_size()

    print_log('Benchmark start.')

    for epoch in range(CONFIG['hyperparameter']['num_epochs']):
        _train(epoch, rank, world_size, train_data, model, criterion, optimizer, lr_scheduler, scaler)
        _test(epoch, rank, world_size, test_data, model, criterion)

    print_log('Benchmark complete.')
