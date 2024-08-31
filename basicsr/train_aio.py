# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
from __future__ import annotations

from collections import defaultdict
from prettytable import PrettyTable
import argparse
import datetime
import logging
import math
import random
import time
import torch
import torch.utils.data
from tqdm import tqdm
from types import SimpleNamespace
from os import path as osp
import os
import json

from basicsr.data import create_dataloader, create_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.datasets import (
    DataType,
    balance_datasets,
    get_test_collate_fn,
    get_train_collate_fn,
    make_dataset,
)
from basicsr.models import create_model
from basicsr.models.aio_model import AIOModel
from basicsr.utils import (
    MessageLogger,
    check_resume,
    get_env_info,
    get_root_logger,
    get_time_str,
    init_tb_logger,
    init_wandb_logger,
    make_exp_dirs,
    mkdir_and_rename,
    set_random_seed,
)
from basicsr.utils.dist_util import get_dist_info, init_dist
from basicsr.utils.options import dict2str, parse


def parse_options(is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--input_path', type=str, required=False, help='The path to the input image. For single image inference only.')
    parser.add_argument('--output_path', type=str, required=False, help='The path to the output image. For single image inference only.')

    args = parser.parse_args()
    opt = parse(args.opt, is_train=is_train)

    # distributed settings
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)
            print('init dist .. ', args.launcher)

    opt['rank'], opt['world_size'] = get_dist_info()

    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    if args.input_path is not None and args.output_path is not None:
        opt['img_path'] = {
            'input_img': args.input_path,
            'output_img': args.output_path
        }

    return opt


def init_loggers(opt):
    log_file = osp.join(opt['path']['log'],
                        f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # initialize wandb logger before tensorboard logger to allow proper sync:
    if (opt['logger'].get('wandb')
            is not None) and (opt['logger']['wandb'].get('project')
                              is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, (
            'should turn on tensorboard when using wandb')
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        # tb_logger = init_tb_logger(log_dir=f'./logs/{opt['name']}') #mkdir logs @CLY
        tb_logger = init_tb_logger(log_dir=osp.join('logs', opt['name']))
    
    return logger, tb_logger


# original
def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loader = None, None
    for phase, dataset_opt in opt['datasets'].items():
        print(phase, dataset_opt)
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = create_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'],
                                            opt['rank'], dataset_enlarge_ratio)
            train_loader = create_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio /
                (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logger.info(
                'Training statistics:'
                f'\n\tNumber of train images: {len(train_set)}'
                f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                f'\n\tWorld size (gpu number): {opt["world_size"]}'
                f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')

        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(
                val_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=None,
                seed=opt['manual_seed'])
            logger.info(
                f'Number of val images/folders in {dataset_opt["name"]}: '
                f'{len(val_set)}')
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loader, total_epochs, total_iters


def make_dataloaders(
    args,
    logger: logging.Logger,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    rank = int(os.environ["RANK"])
    
    data_info = json.load(open(args.data))
    train_datasets, test_datasets = [], []

    data_dict = defaultdict(dict)
    reverse_index = dict()
    for data_name, (data_root, data_type) in data_info.items():
        data_dict[data_name]['type'] = DataType(data_type).name
        data_dict[data_name]['train_ratio'] = 0

        if data_type == DataType.TRAIN_TEST or data_type == DataType.TRAIN_ONLY:
            # For datasets with only train split, we randomly split some train data as test data.
            train_datasets.append(make_dataset(
                data_name, split='train',
                crop_size=args.resolution, overlap=args.overlap,
                train_resize=args.train_resize,
                train_flip_rotate=args.train_flip_rotate,
            ))
            test_datasets.append(make_dataset(
                data_name, split='test',
                crop_size=args.resolution, overlap=args.overlap,
                test_resize=args.test_resize,
            ))
            data_dict[data_name]['train_len'] = len(train_datasets[-1])
            data_dict[data_name]['test_len'] = len(test_datasets[-1])
            data_dict[data_name]['task'] = train_datasets[-1].task
            reverse_index[train_datasets[-1]] = data_name

        elif data_type == DataType.TEST_ONLY:
            # Test-only datasets are used as, well, test datasets.
            test_datasets.append(make_dataset(
                data_name, split='test',
                crop_size=args.resolution, overlap=args.overlap,
                test_resize=args.test_resize,
            ))
            data_dict[data_name]['train_len'] = 0
            data_dict[data_name]['test_len'] = len(test_datasets[-1])
            data_dict[data_name]['task'] = test_datasets[-1].task

        else:
            values = {m.name: m.value for m in DataType}
            raise ValueError(f'unkonw data type {data_type}, expects one of {values}')

    # balance the train datasets to mitigate long-tailed distribution
    balanced_train_datasets = balance_datasets(train_datasets, args.balance_data)
    train_dataset = torch.utils.data.ConcatDataset(balanced_train_datasets)
    test_dataset = torch.utils.data.ConcatDataset(test_datasets)

    # count train_ratio
    for dataset in balanced_train_datasets:
        data_dict[reverse_index[dataset]]['train_ratio'] += 1

    data_table = PrettyTable(
        title=f'Datasets (balance: {args.balance_data})', header=True, align='r',
        field_names=('Name', 'Task', 'Type', '#TrainData', '#BalancedTrainData', '#TestData'),
    )
    for name, info in data_dict.items():
        data_table.add_row([
            name, info['task'], info['type'], info['train_len'], info['train_len'] * info['train_ratio'], info['test_len'],
        ])
    data_table.add_row(['Total', '-', '-', sum(map(len, train_datasets)), len(train_dataset), len(test_dataset)])
    logger.info(f'Datasets Report\n{data_table.get_string()}')

    # TODO: Make samplers savable.
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, seed=args.seed)
    loader_seed = args.seed + rank * args.dataloader_workers
    generator = torch.Generator().manual_seed(loader_seed)
    # NOTE: `accelerator.load_state()` resumes the default rng of random, NumPy, PyTorch,
    # but custom rngs are not affected. So we register them here.
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_workers,
        collate_fn=get_train_collate_fn(args.prompt_strategy),
        pin_memory=True,
        sampler=train_sampler,
        persistent_workers=True,
        generator=generator,
    )

    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        # num_workers=args.dataloader_workers,
        collate_fn=get_test_collate_fn(),
        pin_memory=True,
        sampler=test_sampler,
        # persistent_workers=True,
    )

    return train_loader, test_loader


# ours
def create_train_val_dataloader_aio(opt, logger: logging.Logger):
    args = SimpleNamespace(**opt['args'])
    
    # create train and val dataloaders
    train_loader, val_loader = make_dataloaders(args, logger)
    
    train_loader.dataset.opt = opt['datasets']['train']
    val_loader.dataset.opt = opt['datasets']['val']
    
    total_iters = opt['train']['total_iter']
    total_epochs = total_iters // len(train_loader)
    
    logger.info(
        'Training statistics:'
        f'\n\tNumber of train images: {len(train_loader.dataset)}'
        f'\n\tBatch size per gpu: {args.train_batch_size}'
        f'\n\tWorld size (gpu number): {opt["world_size"]}'
        f'\n\tRequire iter number per epoch: {len(train_loader)}'
        f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')

    return train_loader, train_loader.sampler, val_loader, total_epochs, total_iters


def main():
    # Get distributed training config
    import os
    rank = int(os.environ["RANK"])
    
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=True)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # automatic resume ..
    state_folder_path = 'experiments/{}/training_states/'.format(opt['name'])
    import os
    try:
        states = os.listdir(state_folder_path)
    except BaseException:
        states = []

    resume_state = None
    if len(states) > 0:
        print('!!!!!! resume state .. ', states, state_folder_path)
        max_state_file = '{}.state'.format(max([int(x[0:-6]) for x in states]))
        resume_state = os.path.join(state_folder_path, max_state_file)
        opt['path']['resume_state'] = resume_state

    # load resume states if necessary
    if opt['path'].get('resume_state'):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt['path']['resume_state'],
            map_location=lambda storage, loc: storage.cuda(device_id))
    else:
        resume_state = None

    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt[
                'name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join('tb_logger', opt['name']))

    # initialize loggers
    logger, tb_logger = init_loggers(opt)

    # create train and validation dataloaders
    # result = create_train_val_dataloader(opt, logger)
    result = create_train_val_dataloader_aio(opt, logger)
    train_loader, train_sampler, val_loader, total_epochs, total_iters = result
    # result = create_train_val_dataloader(opt, logger)
    # _, _, val_loader, _, _ = result

    # create model
    if resume_state:  # resume training
        check_resume(opt, resume_state['iter'])
        model: AIOModel = create_model(opt)
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, "
                    f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        model = create_model(opt)
        start_epoch = 0
        current_iter = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # dataloader prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode', None)
    logger.info(f'{prefetch_mode=}')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f'Wrong prefetch_mode {prefetch_mode}.'
                         "Supported ones are: None, 'cuda', 'cpu'.")

    # training
    logger.info(
        f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_time, iter_time = time.time(), time.time()
    start_time = time.time()

    # for epoch in range(start_epoch, total_epochs + 1):
    epoch = start_epoch
    while current_iter <= total_iters:
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        pbar = tqdm(range(len(train_loader)), desc=f'Epoch[{epoch}/{total_epochs}]', disable=(rank != 0))
        while train_data is not None:
            data_time = time.time() - data_time

            current_iter += 1
            if current_iter > total_iters:
                break
            # update learning rate
            model.update_learning_rate(
                current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            # training
            model.feed_data(train_data, is_val=False)
            model.optimize_parameters(current_iter, tb_logger)
            # if result_code == -1 and tb_logger:
            #     print('loss explode .. ')
            #     exit(0)
            iter_time = time.time() - iter_time

            data_time = time.time()
            iter_time = time.time()
            train_data = prefetcher.next()
            pbar.update(1)
        # end of iter
        
        # save models and training states
        if epoch % opt['logger']['save_checkpoint_freq'] == 0:
            logger.info('Saving models and training states.')
            model.save(epoch, current_iter)
        
        # log
        # if current_iter % opt['logger']['print_freq'] == 0:
        log_vars = {'epoch': epoch, 'iter': current_iter, 'total_iter': total_iters}
        log_vars.update({'lrs': model.get_current_learning_rate()})
        log_vars.update({'time': iter_time, 'data_time': data_time})
        log_vars.update(model.get_current_log())
        # print('msg logger .. ', current_iter)
        msg_logger(log_vars)
        
        # validation
        if opt.get('val') is not None and (epoch + 1) % opt['val']['val_freq'] == 0:
            rgb2bgr = opt['val'].get('rgb2bgr', True)
            # wheather use uint8 image to compute metrics
            use_image = opt['val'].get('use_image', True)
            model.validation(val_loader, current_iter, tb_logger,
                                opt['val']['save_img'], rgb2bgr, use_image )
            log_vars = {'epoch': epoch, 'iter': current_iter, 'total_iter': total_iters}
            log_vars.update({'lrs': model.get_current_learning_rate()})
            log_vars.update(model.get_current_log())
            msg_logger(log_vars)
            
        epoch += 1

    # end of epoch

    consumed_time = str(
        datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    if opt.get('val') is not None:
        rgb2bgr = opt['val'].get('rgb2bgr', True)
        use_image = opt['val'].get('use_image', True)
        model.validation(val_loader, current_iter, tb_logger,
                         opt['val']['save_img'], rgb2bgr, use_image)
        # if tb_logger:
        #     print('xxresult! ', opt['name'], ' ', metric)
    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    import os
    os.environ['GRPC_POLL_STRATEGY']='epoll1'
    main()
