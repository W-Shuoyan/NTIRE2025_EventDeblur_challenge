from torch.utils import data as data
from torchvision.transforms.functional import normalize
from tqdm import tqdm
import os
from pathlib import Path
import random
import numpy as np
import torch
import cv2

from basicsr.data.data_util import (paired_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file,
                                    recursive_glob)
from basicsr.data.event_util import events_to_voxel_grid, voxel_norm
from basicsr.data.transforms import augment, triple_random_crop, random_augmentation
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, get_root_logger
from torch.utils.data.dataloader import default_collate

def increase_brightness_cv(image, brightness_factor=2):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    enhanced_image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)
    return enhanced_image.astype(np.float32) / 255.0


class HighREV_train_dataset(data.Dataset):
    """Paired npz and png dataset for event-based single image deblurring.
    --HighREV
    |----train
    |    |----blur
    |    |    |----SEQNAME_%5d.png
    |    |    |----...
    |    |----event
    |    |    |----SEQNAME_%5d_%2d.npz
    |    |    |----...
    |    |----sharp
    |    |    |----SEQNAME_%5d.png
    |    |    |----...
    |----val
    ...
    """

    def __init__(self, opt):
        super(HighREV_train_dataset, self).__init__()
        self.opt = opt
        self.dataroot = Path(opt['dataroot'])
        self.moments = opt['moment_events']

        self.split = 'train' if opt['phase'] == 'train' else 'val'
        self.norm_voxel = opt['norm_voxel']
        self.dataPath = []

        blur_frames = sorted(recursive_glob(rootdir=os.path.join(self.dataroot, self.split, 'blur'), suffix='.png'))
        blur_frames = [os.path.join(self.dataroot, self.split, 'blur', blur_frame) for blur_frame in blur_frames]
        
        sharp_frames = sorted(recursive_glob(rootdir=os.path.join(self.dataroot, self.split, 'sharp'), suffix='.png'))
        sharp_frames = [os.path.join(self.dataroot, self.split, 'sharp', sharp_frame) for sharp_frame in sharp_frames]

        event_frames = sorted(recursive_glob(rootdir=os.path.join(self.dataroot, self.split, 'event'), suffix='.npz'))
        event_frames = [os.path.join(self.dataroot, self.split, 'event', event_frame) for event_frame in event_frames]
        
        assert len(blur_frames) == len(sharp_frames), f"Mismatch in blur ({len(blur_frames)}) and sharp ({len(sharp_frames)}) frame counts."

        for i in range(len(blur_frames)):
            blur_name = os.path.basename(blur_frames[i])  # e.g., SEQNAME_00001.png
            base_name = os.path.splitext(blur_name)[0]   # Remove .png, get SEQNAME_00001
            event_list = sorted([f for f in event_frames if f.startswith(os.path.join(self.dataroot, self.split, 'event', base_name + '_'))])[:-1] # remove the last one because it is out of the exposure time

            self.dataPath.append({
                'blur_path': blur_frames[i],
                'sharp_path': sharp_frames[i],
                'event_paths': event_list
            })

        self.file_client = None
        self.io_backend_opt = opt['io_backend']


    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        image_path = self.dataPath[index]['blur_path']
        gt_path = self.dataPath[index]['sharp_path']
        event_paths = self.dataPath[index]['event_paths']


        # get LQ
        img_bytes = self.file_client.get(image_path)  # 'lq'
        img_lq = imfrombytes(img_bytes, float32=True)
        # get GT
        img_bytes = self.file_client.get(gt_path)    # 'gt'
        img_gt = imfrombytes(img_bytes, float32=True)

        h_img, w_img, _ = img_lq.shape

        if 'lq_size' in self.opt:
            lq_size = self.opt['lq_size']
        else:
            lq_size = img_lq.shape[:-1]

        ## Read event and convert to voxel grid:
        events = [np.load(event_path) for event_path in event_paths]

        events = np.concatenate([
            np.column_stack((event['timestamp'], event['y'], event['x'], event['polarity'])).astype(np.float32)
            for event in events], axis=0)
        voxels = events_to_voxel_grid(events, num_bins=self.moments+1, width=w_img, height=h_img,
                                        return_format='HWC')
        
        h_lq, w_lq = lq_size
        h0 = random.randint(0, h_img - h_lq)
        w0 = random.randint(0, w_img - w_lq)
        
        img_lq = img_lq[h0:h0 + h_lq, w0:w0 + w_lq, :]
        img_gt = img_gt[h0:h0 + h_lq, w0:w0 + w_lq, :]
        voxels = voxels[h0:h0 + h_lq, w0:w0 + w_lq, :]

        tensors = [img_lq, img_gt, voxels]
        tensors = augment(tensors, True, True)
        tensors = img2tensor(tensors)
        img_lq, img_gt, voxels = tensors

        if self.norm_voxel:
            voxels = voxel_norm(voxels)
        
        all_voxel = []
        for i in range(voxels.shape[0]-1):
            all_voxel.append(voxels[i:i+2, :, :])
        voxels = torch.stack(all_voxel, dim=0)

        origin_index = os.path.basename(image_path).split('.')[0]

        return {'frame': img_lq, 'frame_gt': img_gt, 'voxel': voxels, 'image_name': origin_index}


    def __len__(self):
        return len(self.dataPath)


class HighREV_finetune_dataset(data.Dataset):
    """Paired npz and png dataset for event-based single image deblurring.
    --HighREV
    |----train
    |    |----blur
    |    |    |----sternwatz_window_%5d.png
    |    |    |----...
    |    |----event
    |    |    |----sternwatz_window_%5d_%2d.npz
    |    |    |----...
    |    |----sharp
    |    |    |----sternwatz_window_%5d.png
    |    |    |----...
    |----val
    ...
    """

    def __init__(self, opt):
        super(HighREV_finetune_dataset, self).__init__()
        self.opt = opt
        self.dataroot = Path(opt['dataroot'])
        self.moments = opt['moment_events']

        self.split = 'train' if opt['phase'] == 'train' else 'val'
        self.norm_voxel = opt['norm_voxel']
        self.dataPath = []

        blur_frames = sorted(recursive_glob(rootdir=os.path.join(self.dataroot, self.split, 'blur'), suffix='.png'))
        blur_frames = [f for f in blur_frames if os.path.basename(f).startswith("sternwatz_window")]
        blur_frames = [os.path.join(self.dataroot, self.split, 'blur', blur_frame) for blur_frame in blur_frames]
        
        sharp_frames = sorted(recursive_glob(rootdir=os.path.join(self.dataroot, self.split, 'sharp'), suffix='.png'))
        sharp_frames = [f for f in sharp_frames if os.path.basename(f).startswith("sternwatz_window")]
        sharp_frames = [os.path.join(self.dataroot, self.split, 'sharp', sharp_frame) for sharp_frame in sharp_frames]

        event_frames = sorted(recursive_glob(rootdir=os.path.join(self.dataroot, self.split, 'event'), suffix='.npz'))
        event_frames = [f for f in event_frames if os.path.basename(f).startswith("sternwatz_window")]
        event_frames = [os.path.join(self.dataroot, self.split, 'event', event_frame) for event_frame in event_frames]
        
        assert len(blur_frames) == len(sharp_frames), f"Mismatch in blur ({len(blur_frames)}) and sharp ({len(sharp_frames)}) frame counts."

        for i in range(len(blur_frames)):
            blur_name = os.path.basename(blur_frames[i])  # e.g., SEQNAME_00001.png
            base_name = os.path.splitext(blur_name)[0]   # Remove .png, get SEQNAME_00001
            event_list = sorted([f for f in event_frames if f.startswith(os.path.join(self.dataroot, self.split, 'event', base_name + '_'))])[:-1] # remove the last one because it is out of the exposure time

            self.dataPath.append({
                'blur_path': blur_frames[i],
                'sharp_path': sharp_frames[i],
                'event_paths': event_list
            })

        self.file_client = None
        self.io_backend_opt = opt['io_backend']


    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        image_path = self.dataPath[index]['blur_path']
        gt_path = self.dataPath[index]['sharp_path']
        event_paths = self.dataPath[index]['event_paths']


        # get LQ
        img_bytes = self.file_client.get(image_path)  # 'lq'
        img_lq = imfrombytes(img_bytes, float32=True)
        # get GT
        img_bytes = self.file_client.get(gt_path)    # 'gt'
        img_gt = imfrombytes(img_bytes, float32=True)

        img_gt_cv = (img_gt * 255).astype(np.uint8)
        img_lq_cv = (img_lq * 255).astype(np.uint8)

        img_gt_cv = increase_brightness_cv(img_gt_cv)
        img_lq_cv = increase_brightness_cv(img_lq_cv)

        img_gt = img_gt_cv.astype(np.float32)
        img_lq = img_lq_cv.astype(np.float32)
        
        h_img, w_img, _ = img_lq.shape

        if 'lq_size' in self.opt:
            lq_size = self.opt['lq_size']
        else:
            lq_size = img_lq.shape[:-1]

        ## Read event and convert to voxel grid:
        events = [np.load(event_path) for event_path in event_paths]

        events = np.concatenate([
            np.column_stack((event['timestamp'], event['y'], event['x'], event['polarity'])).astype(np.float32)
            for event in events], axis=0)
        voxels = events_to_voxel_grid(events, num_bins=self.moments+1, width=w_img, height=h_img,
                                        return_format='HWC')
        
        h_lq, w_lq = lq_size
        h0 = random.randint(0, h_img - h_lq)
        w0 = random.randint(0, w_img - w_lq)
        
        img_lq = img_lq[h0:h0 + h_lq, w0:w0 + w_lq, :]
        img_gt = img_gt[h0:h0 + h_lq, w0:w0 + w_lq, :]
        voxels = voxels[h0:h0 + h_lq, w0:w0 + w_lq, :]

        tensors = [img_lq, img_gt, voxels]
        tensors = augment(tensors, True, True)
        tensors = img2tensor(tensors)
        img_lq, img_gt, voxels = tensors

        if self.norm_voxel:
            voxels = voxel_norm(voxels)
        
        all_voxel = []
        for i in range(voxels.shape[0]-1):
            all_voxel.append(voxels[i:i+2, :, :])
        voxels = torch.stack(all_voxel, dim=0)

        origin_index = os.path.basename(image_path).split('.')[0]

        return {'frame': img_lq, 'frame_gt': img_gt, 'voxel': voxels, 'image_name': origin_index}


    def __len__(self):
        return len(self.dataPath)


class HighREV_test_dataset(data.Dataset):
    def __init__(self, opt):
        super(HighREV_test_dataset, self).__init__()
        self.opt = opt
        self.dataroot = Path(opt['dataroot'])
        self.moments = opt['moment_events']

        self.norm_voxel = opt['norm_voxel']
        self.dataPath = []

        blur_frames = sorted(recursive_glob(rootdir=os.path.join(self.dataroot, 'blur'), suffix='.png'))
        blur_frames = [os.path.join(self.dataroot, 'blur', blur_frame) for blur_frame in blur_frames]

        event_frames = sorted(recursive_glob(rootdir=os.path.join(self.dataroot, 'event'), suffix='.npz'))
        event_frames = [os.path.join(self.dataroot, 'event', event_frame) for event_frame in event_frames]
        
        for i in range(len(blur_frames)):
            blur_name = os.path.basename(blur_frames[i])  # e.g., SEQNAME_00001.png
            base_name = os.path.splitext(blur_name)[0]   # Remove .png, get SEQNAME_00001
            event_list = sorted([f for f in event_frames if f.startswith(os.path.join(self.dataroot, 'event', base_name + '_'))])[:-1] # remove the last one because it is out of the exposure time

            self.dataPath.append({
                'blur_path': blur_frames[i],
                'event_paths': event_list
            })

        self.file_client = None
        self.io_backend_opt = opt['io_backend']


    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        image_path = self.dataPath[index]['blur_path']
        event_paths = self.dataPath[index]['event_paths']


        # get LQ
        img_bytes = self.file_client.get(image_path)  # 'lq'
        img_lq = imfrombytes(img_bytes, float32=True)

        h_img, w_img, _ = img_lq.shape

        if 'lq_size' in self.opt:
            lq_size = self.opt['lq_size']
        else:
            lq_size = img_lq.shape[:-1]

        ## Read event and convert to voxel grid:
        events = [np.load(event_path) for event_path in event_paths]

        events = np.concatenate([
            np.column_stack((event['timestamp'], event['y'], event['x'], event['polarity'])).astype(np.float32)
            for event in events], axis=0)
        voxels = events_to_voxel_grid(events, num_bins=self.moments+1, width=w_img, height=h_img,
                                        return_format='HWC')
        
        h_lq, w_lq = lq_size
        h0 = random.randint(0, h_img - h_lq)
        w0 = random.randint(0, w_img - w_lq)
        
        img_lq = img_lq[h0:h0 + h_lq, w0:w0 + w_lq, :]
        voxels = voxels[h0:h0 + h_lq, w0:w0 + w_lq, :]

        tensors = [img_lq, voxels]
        # tensors = augment(tensors, True, True)
        tensors = img2tensor(tensors)
        img_lq, voxels = tensors

        if self.norm_voxel:
            voxels = voxel_norm(voxels)
        
        all_voxel = []
        for i in range(voxels.shape[0]-1):
            all_voxel.append(voxels[i:i+2, :, :])
        voxels = torch.stack(all_voxel, dim=0)

        origin_index = os.path.basename(image_path).split('.')[0]

        return {'frame': img_lq, 'voxel': voxels, 'image_name': origin_index}


    def __len__(self):
        return len(self.dataPath)