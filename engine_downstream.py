# Original work Copyright (c) Meta Platforms, Inc. and affiliates. <https://github.com/facebookresearch/mae>
# Modified work Copyright 2026 Zijie Zhu . <https://github.com/tigerforest0573/Kust_MeAI2025>

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# ST-MEM: https://github.com/bakqui/ST-MEM
# https://github.com/physionetchallenges/python-example-2025
# --------------------------------------------------------

import math
import sys
from typing import Dict, Iterable, Optional, Tuple

import torch
import torchmetrics

import util.misc as misc
import util.lr_sched as lr_sched
from challenge_score import calculate_challenge_score
import torch.nn.functional as F
def train_one_epoch(model: torch.nn.Module,
                    criterion: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    log_writer=None,
                    config: Optional[dict] = None,
                    use_amp: bool = True,
                    ) -> Dict[str, float]:
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = config['accum_iter']
    max_norm = config['max_norm']

    optimizer.zero_grad()
    #用于计算challenge_score
    all_train_preds = []
    all_train_targets = []
    if log_writer is not None:
        print(f'log_dir: {log_writer.log_dir}')

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, config)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(samples)
            targets_one_hot = F.one_hot(targets, num_classes=2).float()
            loss = criterion(outputs, targets_one_hot)
            # loss = criterion(outputs, targets)
        #计算challenge score
        with torch.no_grad():
            #获取class 1的概率
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_train_preds.append(probs.detach())
            all_train_targets.append(targets.detach())

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss,
                    optimizer,
                    clip_grad=max_norm,
                    parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]['lr']
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((epoch + data_iter_step / len(data_loader)) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
    local_preds = torch.cat(all_train_preds)
    local_targets = torch.cat(all_train_targets)
    # print(f"DEBUG Local Preds Length: {local_preds.shape[0]}")
    # print(f"DEBUG Local Targets Length: {local_targets.shape[0]}")
    # 从所有 GPU 收集数据 (DDP)
    global_preds = misc.concat_all_gather(local_preds)
    global_targets = misc.concat_all_gather(local_targets)
    # print(f"DEBUG Global Preds Length: {global_preds.shape[0]}")
    # print(f"DEBUG Global Targets Length: {global_targets.shape[0]}")
    np_preds = global_preds.cpu().numpy()
    np_targets = global_targets.cpu().numpy()
        
    challenge_score = calculate_challenge_score(np_preds, np_targets, alpha=0.05)
    #记录到Tensorboard和 Log
    if log_writer is not None:
         log_writer.add_scalar('train_challenge_score', challenge_score, epoch)
    print(f"Epoch {epoch} Training Challenge Score: {challenge_score:.4f}")

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module,
             criterion: torch.nn.Module,
             data_loader: Iterable,
             device: torch.device,
             metric_fn: torchmetrics.Metric,
             output_act: torch.nn.Module,
             target_dtype: torch.dtype = torch.long,
             use_amp: bool = True,
             ) -> Tuple[Dict[str, float], Dict[str, float]]:
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    #用于储存challenge_score
    all_preds_list = []
    all_targets_list = []

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            if samples.ndim == 4:  # batch_size, n_drops, n_channels, n_frames
                logits_list = []
                for i in range(samples.size(1)):
                    logits = model(samples[:, i])
                    logits_list.append(logits)
                logits_list = torch.stack(logits_list, dim=1)
                outputs_list = output_act(logits_list)
                logits = logits_list.mean(dim=1)
                outputs = outputs_list.mean(dim=1)
            else:
                logits = model(samples)
                outputs = output_act(logits)
            targets_one_hot = F.one_hot(targets, num_classes=2).float()
            loss = criterion(logits, targets_one_hot)
            # loss = criterion(logits, targets)
            

        outputs = misc.concat_all_gather(outputs)
        targets = misc.concat_all_gather(targets).to(dtype=target_dtype)
        metric_fn.update(outputs, targets)
        metric_logger.meters['loss'].update(loss.item(), n=samples.size(0))
        #取1
        pos_probs = outputs[:, 1]
        all_preds_list.append(pos_probs.cpu())
        all_targets_list.append(targets.cpu())

    metric_logger.synchronize_between_processes()
    valid_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    final_preds = torch.cat(all_preds_list).numpy()
    final_targets = torch.cat(all_targets_list).numpy()
    
    # 计算 
    
    val_challenge_score = calculate_challenge_score(final_preds, final_targets, alpha=0.05)
    print(f" EVAL Challenge Score: {val_challenge_score:.4f}")
    metrics = metric_fn.compute()
    if isinstance(metrics, dict):  # MetricCollection
        metrics = {k: v.item() for k, v in metrics.items()}
    else:
        metrics = {metric_fn.__class__.__name__: metrics.item()}
    metrics['challenge_score'] = val_challenge_score
    metric_str = "  ".join([f"{k}: {v:.3f}" for k, v in metrics.items()])
    metric_str = f"{metric_str} loss: {metric_logger.loss.global_avg:.3f}"
    print(f"* {metric_str}")
    metric_fn.reset()

    return valid_stats, metrics
