# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
from collections import defaultdict
import importlib
import itertools
import numpy as np
import torch
import torch.distributed
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
from prettytable import PrettyTable

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.dist_util import get_dist_info
from basicsr.utils.toolkit import AverageMeter, compute_psnr_ssim, gather_together

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')


class AIOModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(AIOModel, self).__init__(opt)

        # define network
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

        self.scale = int(opt['scale'])

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            percep_type = train_opt['perceptual_opt'].pop('type')
            cri_perceptual_cls = getattr(loss_module, percep_type)
            self.cri_perceptual = cri_perceptual_cls(
                **train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
        #         if k.startswith('module.offsets') or k.startswith('module.dcns'):
        #             optim_params_lowlr.append(v)
        #         else:
                optim_params.append(v)
            # else:
            #     logger = get_root_logger()
            #     logger.warning(f'Params {k} will not be optimized.')
        # print(optim_params)
        # ratio = 0.1

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam([{'params': optim_params}],
                                                **train_opt['optim_g'])
        elif optim_type == 'SGD':
            self.optimizer_g = torch.optim.SGD(optim_params,
                                               **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW([{'params': optim_params}],
                                                **train_opt['optim_g'])
            pass
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data, is_val=False):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def grids(self):
        b, c, h, w = self.gt.size()
        self.original_size = (b, c, h, w)

        assert b == 1
        if 'crop_size_h' in self.opt['val']:
            crop_size_h = self.opt['val']['crop_size_h']
        else:
            crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)

        if 'crop_size_w' in self.opt['val']:
            crop_size_w = self.opt['val'].get('crop_size_w')
        else:
            crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)


        crop_size_h, crop_size_w = crop_size_h // self.scale * self.scale, crop_size_w // self.scale * self.scale
        #adaptive step_i, step_j
        num_row = (h - 1) // crop_size_h + 1
        num_col = (w - 1) // crop_size_w + 1

        import math
        step_j = crop_size_w if num_col == 1 else math.ceil((w - crop_size_w) / (num_col - 1) - 1e-8)
        step_i = crop_size_h if num_row == 1 else math.ceil((h - crop_size_h) / (num_row - 1) - 1e-8)

        scale = self.scale
        step_i = step_i//scale*scale
        step_j = step_j//scale*scale

        parts = []
        idxes = []

        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + crop_size_h >= h:
                i = h - crop_size_h
                last_i = True

            last_j = False
            while j < w and not last_j:
                if j + crop_size_w >= w:
                    j = w - crop_size_w
                    last_j = True
                parts.append(self.lq[:, :, i // scale :(i + crop_size_h) // scale, j // scale:(j + crop_size_w) // scale])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        self.origin_lq = self.lq
        self.lq = torch.cat(parts, dim=0)
        self.idxes = idxes

    def grids_inverse(self):
        preds = torch.zeros(self.original_size)
        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w))
        if 'crop_size_h' in self.opt['val']:
            crop_size_h = self.opt['val']['crop_size_h']
        else:
            crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)

        if 'crop_size_w' in self.opt['val']:
            crop_size_w = self.opt['val'].get('crop_size_w')
        else:
            crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)

        crop_size_h, crop_size_w = crop_size_h // self.scale * self.scale, crop_size_w // self.scale * self.scale

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            preds[0, :, i: i + crop_size_h, j: j + crop_size_w] += self.outs[cnt]
            count_mt[0, 0, i: i + crop_size_h, j: j + crop_size_w] += 1.

        self.output = (preds / count_mt).to(self.device)
        self.lq = self.origin_lq

    def optimize_parameters(self, current_iter, tb_logger):
        self.optimizer_g.zero_grad()

        if self.opt['train'].get('mixup', False):
            self.mixup_aug()

        preds = self.net_g(self.lq)
        if not isinstance(preds, list):
            preds = [preds]

        self.output = preds[-1]

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = 0.
            for pred in preds:
                l_pix += self.cri_pix(pred, self.gt)

            # print('l pix ... ', l_pix)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
        #
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style


        l_total = l_total + 0. * sum(p.sum() for p in self.net_g.parameters())

        l_total.backward()
        use_grad_clip = self.opt['train'].get('use_grad_clip', True)
        if use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        self.net_g.eval()
        # print(f'{self.lq.shape=}, {self.gt.shape=}')
        with torch.no_grad():
            n = len(self.lq)
            outs = []
            m = self.opt['val'].get('max_minibatch', n)
            assert m == 1, f'Currently, we expect batchsize to be 1; gets {m}'
            i = 0
            while i < n:
                j = i + m
                if j >= n:
                    j = n
                pred = self.net_g(self.lq[i:j])
                if isinstance(pred, list):
                    pred = pred[-1]
                outs.append(pred.detach().cpu())
                i = j

            self.output = torch.cat(outs, dim=0)
        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            ssim_loss = AverageMeter('ssim', ':.4f')
            psnr_loss = AverageMeter('psnr', ':.4f')
            eval_res = defaultdict(dict)  # dataset_name => task => metric_name => [per_sample_metric_values]

        rank, world_size = get_dist_info()
        if rank == 0:
            pbar = tqdm(total=len(dataloader), unit='image')

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            if idx % world_size != rank:
                continue
            lq_path = val_data['lq_path'][0]
            img_name = osp.splitext(osp.basename(lq_path))[0]
            
            try:
                self.feed_data(val_data, is_val=True)
                if self.opt['val'].get('grids', False):
                    self.grids()

                self.test()

                if self.opt['val'].get('grids', False):
                    self.grids_inverse()

                visuals = self.get_current_visuals()
                sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
                if 'gt' in visuals:
                    gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                    del self.gt

                # tentative for out of GPU memory
                del self.lq
                del self.output
                torch.cuda.empty_cache()

                if save_img:
                    if sr_img.shape[2] == 6:
                        L_img = sr_img[:, :, :3]
                        R_img = sr_img[:, :, 3:]

                        # visual_dir = osp.join('visual_results', dataset_name, self.opt['name'])
                        visual_dir = osp.join(self.opt['path']['visualization'], dataset_name)

                        imwrite(L_img, osp.join(visual_dir, f'{img_name}_L.png'))
                        imwrite(R_img, osp.join(visual_dir, f'{img_name}_R.png'))
                    else:
                        if self.opt['is_train']:

                            save_img_path = osp.join(self.opt['path']['visualization'],
                                                    img_name,
                                                    f'{img_name}_{current_iter}.png')

                            save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                                    img_name,
                                                    f'{img_name}_{current_iter}_gt.png')
                        else:
                            save_img_path = osp.join(
                                self.opt['path']['visualization'], dataset_name,
                                f'{img_name}.png')
                            save_gt_img_path = osp.join(
                                self.opt['path']['visualization'], dataset_name,
                                f'{img_name}_gt.png')

                        imwrite(sr_img, save_img_path)
                        imwrite(gt_img, save_gt_img_path)

                if with_metrics:
                    # calculate metrics
                    psnr, ssim = compute_psnr_ssim(sr_img, gt_img)
                    psnr_loss.update(psnr)
                    ssim_loss.update(ssim)
                    dataset = val_data['dataset'][0]
                    subset = val_data['subset'][0]
                    eval_res[dataset].setdefault(subset, {'SSIM': [], 'PSNR': []})
                    eval_res[dataset][subset]['SSIM'].append(ssim)
                    eval_res[dataset][subset]['PSNR'].append(psnr)

                cnt += 1
                if rank == 0:
                    for _ in range(world_size):
                        pbar.update(1)
                        pbar.set_description(f'Test {img_name}')
                        pbar.set_postfix(SSIM=ssim_loss.format_values(), PSNR=psnr_loss.format_values())

            except BaseException:
                raise RuntimeError(f'Exception during processing {lq_path}')

        torch.cuda.empty_cache()

        # dataset => subset => metric_name => [[per_sample_metric_values] for ranks]
        eval_ssim_psnr: dict[str, dict[str, dict[str, list[list[float]]]]] = {}
        # for sublist in gather_objects(args, accelerator, eval_res):
        for sublist in gather_together(eval_res):
            for dataset, v in sublist.items(): # v: dataset => subset => metric_name => metric_values
                eval_ssim_psnr.setdefault(dataset, dict())
                for subset, sub_v in v.items():  # sub_v: subset => metric_name => metric_values
                    eval_ssim_psnr[dataset].setdefault(subset, {'SSIM': [], 'PSNR': []})
                    eval_ssim_psnr[dataset][subset]['SSIM'].append(sub_v['SSIM'])
                    eval_ssim_psnr[dataset][subset]['PSNR'].append(sub_v['PSNR'])
                    
        if rank == 0:
            logger = get_root_logger()
            metric_table = PrettyTable(
                title='Metrics mean(min/max)', header=True, align='c',
                field_names=('Dataset', 'Subset', 'SSIM', 'PSNR'),
            )
            logs = dict()
            # Log detailed metrics.
            for dataset, v in eval_ssim_psnr.items():  # v: subset => metric_name => [[per_sample_metric_values] for ranks]
                for subset, sub_v in v.items():  # sub_v: metric_name => [[per_sample_metric_values] for ranks]
                    ssim, psnr = sub_v['SSIM'], sub_v['PSNR']
                    all_ssim, all_psnr = np.array(list(itertools.chain(*ssim))), np.array(list(itertools.chain(*psnr)))
                    ssim_str = f'{all_ssim.mean():.3f}({all_ssim.min():.3f}/{all_ssim.max():.3f})'
                    psnr_str = f'{all_psnr.mean():.3f}({all_psnr.min():.3f}/{all_psnr.max():.3f})'
                    metric_table.add_row([dataset, subset, ssim_str, psnr_str])
                    logs[f'm_{dataset}-{subset}-SSIM'] = all_ssim.mean()
                    logs[f'm_{dataset}-{subset}-PSNR'] = all_psnr.mean()
            logger.info(f'Evaluation Report\n{metric_table.get_string()}')
            self.log_dict = logs
        torch.distributed.barrier()
        
        return 0.

    def nondist_validation(self, *args, **kwargs):
        logger = get_root_logger()
        logger.warning('nondist_validation is not implemented. Run dist_validation.')
        self.dist_validation(*args, **kwargs)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        try:
            self.save_network(self.net_g, 'net_g', current_iter)
            self.save_training_state(epoch, current_iter)
        except BaseException as e:
            print(f'Error saving checkpoint at {epoch=}, {current_iter=}: {e}')
