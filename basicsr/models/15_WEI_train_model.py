import importlib
import torch
from torch.nn import functional as F
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
import os
import math
import time
import logging
from collections import OrderedDict

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img, get_model_flops
from basicsr.metrics import calculate_psnr, calculate_ssim

loss_module = importlib.import_module('basicsr.models.losses')
logger = logging.getLogger('basicsr')


class WEI_train_model(BaseModel):
    def __init__(self, opt):
        super(WEI_train_model, self).__init__(opt)

        # define network
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # flops
        if self.opt.get('print_flops', False):
            input_dim = self.opt.get('flops_input_shape', [(3, 256, 256),(9, 2, 256, 256)])
            flops = get_model_flops(self.net_g, input_dim, False)
            flops = flops/10**9
            logger.info("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))



        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True),
                              param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        # define losses
        if train_opt.get('pixel_opt'):
            self.pixel_type = train_opt['pixel_opt'].pop('type')
            # print('LOSS: pixel_type:{}'.format(self.pixel_type))
            cri_pix_cls = getattr(loss_module, self.pixel_type)

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
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['frame'].to(self.device)
        self.voxel=data['voxel'].to(self.device) 
        if 'frame_gt' in data:
            self.gt = data['frame_gt'].to(self.device)
        if 'image_name' in data:
            self.image_name = data['image_name'][0]


    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(image=self.lq, event=self.voxel)
        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        use_grad_clip = self.opt['train'].get('use_grad_clip', True)
        if use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            start_time = time.time()
            self.pre_process()
            if 'tile' in self.opt:
                    self.tile_process()
            else:
                    self.process()
            self.post_process()
            end_time = time.time()
            self.running_time = end_time - start_time
        self.net_g.train()

    def single_image_inference(self, img, voxel, save_path):
        self.feed_data(data={'frame': img.unsqueeze(dim=0), 'voxel': voxel.unsqueeze(dim=0)})
        if self.opt['val'].get('grids') is not None:
            self.grids()
            self.grids_voxel()

        self.test()

        if self.opt['val'].get('grids') is not None:
            self.grids_inverse()
            # self.grids_inverse_voxel()

        visuals = self.get_current_visuals()
        sr_img = tensor2img([visuals['result']])
        imwrite(sr_img, save_path)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        get_root_logger()
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img)
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        dataset_name = self.opt.get('name')
        save_gt = self.opt['val'].get('save_gt', False)
        with_metrics = self.opt['val'].get('cal_metrics', True)

        psnr = []
        ssim = []
        all_time = []

        pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            self.feed_data(val_data)
            self.test()
            all_time.append(self.running_time)
            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{self.image_name}.png')
                save_gt_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{self.image_name}_gt.png')
                imwrite(sr_img, save_img_path)
                if save_gt:
                        imwrite(gt_img, save_gt_img_path)

                if with_metrics:
                    # calculate metrics
                    psnr_values = calculate_psnr(
                        sr_img, gt_img, 0, test_y_channel=True)
                    ssim_values = calculate_ssim(
                        sr_img, gt_img, 0, test_y_channel=True)
                    psnr.append(psnr_values)
                    ssim.append(ssim_values)
        
            pbar.update(1)
            pbar.set_description(f'Test {self.image_name}')

        pbar.close()
        if with_metrics:
            self.avg_psnr = torch.mean(torch.tensor(psnr))
            self.avg_ssim = torch.mean(torch.tensor(ssim))
            self.avg_time = torch.mean(torch.tensor(all_time))
            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name},\t'
        log_str += f'\t # PSNR: {self.avg_psnr:.4f}'
        log_str += f'\t # SSIM: {self.avg_ssim:.4f}'
        log_str += f'\t # running_time: {self.avg_time:.4f} s'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            tb_logger.add_scalar(f'metrics/PSNR',
                                 self.avg_psnr, current_iter)
            tb_logger.add_scalar(f'metrics/SSIM',
                                 self.avg_ssim, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

    def pre_process(self):
        window_size = self.opt['val'].get('window_size', 4)
        self.mod_pad_h, self.mod_pad_w = 0, 0
        _, _, h, w = self.lq.shape
        if h % window_size != 0:
            self.mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            self.mod_pad_w = window_size - w % window_size
        self.lq_pad = F.pad(self.lq, (0, self.mod_pad_w, 0, self.mod_pad_h), 'replicate')
        self.voxel_pad = F.pad(self.voxel, (0, self.mod_pad_w, 0, self.mod_pad_h, 0, 0), 'replicate')

    def process(self):
        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(image=self.lq, event=self.voxel)

    def post_process(self):
        _, _, h, w = self.output.shape
        output_height = h - self.mod_pad_h
        output_width = w - self.mod_pad_w
        self.output = self.output[:, :, 0:output_height, 0:output_width]

    def tile_process(self):
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/ata4/esrgan-launcher
        """
        batch, channel, height, width = self.lq_pad.shape
        output_height = height
        output_width = width
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        self.output = self.lq_pad.new_zeros(output_shape)
        tiles_x = math.ceil(width / self.opt['tile']['tile_size'])
        tiles_y = math.ceil(height / self.opt['tile']['tile_size'])

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * self.opt['tile']['tile_size']
                ofs_y = y * self.opt['tile']['tile_size']
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.opt['tile']['tile_size'], width)
                input_start_y = ofs_y
                input_end_y = min(
                    ofs_y + self.opt['tile']['tile_size'], height)

                # input tile area on total image with padding
                input_start_x_pad = max(
                    input_start_x - self.opt['tile']['tile_pad'], 0)
                input_end_x_pad = min(
                    input_end_x + self.opt['tile']['tile_pad'], width)
                input_start_y_pad = max(
                    input_start_y - self.opt['tile']['tile_pad'], 0)
                input_end_y_pad = min(
                    input_end_y + self.opt['tile']['tile_pad'], height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                img_tile = self.lq_pad[:, :, input_start_y_pad:input_end_y_pad,
                                   input_start_x_pad:input_end_x_pad]
                voxel_tile = self.voxel_pad[:, :, :, input_start_y_pad:input_end_y_pad,
                                     input_start_x_pad:input_end_x_pad]

                # upscale tile
                try:
                    self.net_g.eval()
                    with torch.no_grad():
                        output_tile = self.net_g(image=img_tile, event=voxel_tile)
                except RuntimeError as error:
                    print('Error', error)

                # output tile area on total image
                output_start_x = input_start_x
                output_end_x = input_end_x
                output_start_y = input_start_y
                output_end_y = input_end_y

                # output tile area without padding
                output_start_x_tile = input_start_x - input_start_x_pad
                output_end_x_tile = output_start_x_tile + input_tile_width
                output_start_y_tile = input_start_y - input_start_y_pad
                output_end_y_tile = output_start_y_tile + input_tile_height

                # put tile into output image
                self.output[:, :, output_start_y:output_end_y,
                         output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                                                    output_start_x_tile:output_end_x_tile]
