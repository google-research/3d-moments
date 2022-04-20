# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from data_loaders.data_utils import (resize_img, resize_img_intrinsic, parse_pose_file, unnormalize_intrinsics)
import torchvision
from core.utils import remove_noise_in_dpt_disparity


class MannequinChallengeDataset(Dataset):
    def __init__(self, args, subset, **kwargs):
        base_dir = '/mnt/filestore/Mannequin4000/2021-08-04/'
        if subset == 'val' or subset == 'validation':
            subset = 'test'
        valid_sequence_list_file = os.path.join(base_dir, 'valid', '{}.txt'.format(subset))
        self.img_dir = os.path.join(base_dir, 'images')
        self.depth_dir = os.path.join(base_dir, 'dpt')
        self.scale_shift_dir = os.path.join(base_dir, 'scale_shift')
        self.camera_files = np.genfromtxt(valid_sequence_list_file, dtype='str')
        self.to_tensor = torchvision.transforms.ToTensor()
        self.ds_factor = 0.4
        self.interval = 16

    def __len__(self):
        return len(self.camera_files)

    def __getitem__(self, idx):
        camera_file = self.camera_files[idx]
        video_id, cameras = parse_pose_file(camera_file)
        timestamps = list(cameras.keys())
        num_frames = len(timestamps)
        interval = max(1, num_frames - self.interval)
        id1 = np.random.choice(interval)
        id2 = min(id1 + np.random.randint(1, self.interval), num_frames - 1)
        if np.random.choice([0, 1]):
            id1, id2 = id2, id1
        timestamp1 = timestamps[id1]
        timestamp2 = timestamps[id2]

        img_file1 = os.path.join(self.img_dir, video_id, '{}_{}.jpg'.format(video_id, timestamp1))
        img_file2 = os.path.join(self.img_dir, video_id, '{}_{}.jpg'.format(video_id, timestamp2))
        image1 = imageio.imread(img_file1) / 255.
        image2 = imageio.imread(img_file2) / 255.

        disp1 = imageio.imread(os.path.join(self.depth_dir, video_id, '{}_{}.png'.format(video_id, timestamp1))) / 65535.
        disp2 = imageio.imread(os.path.join(self.depth_dir, video_id, '{}_{}.png'.format(video_id, timestamp2))) / 65535.
        disp1 = remove_noise_in_dpt_disparity(disp1)
        disp2 = remove_noise_in_dpt_disparity(disp2)

        scale_shift_file = os.path.join(self.scale_shift_dir, os.path.basename(camera_file))
        scale_shifts = np.loadtxt(scale_shift_file)

        scale_shift1 = scale_shifts[id1]
        scale_shift2 = scale_shifts[id2]

        depth1 = 1. / np.maximum(disp1, 0.01)
        depth2 = 1. / np.maximum(disp2, 0.01)

        # depth1 = sparse_bilateral_filtering(depth1, num_iter=1)  # fixme
        # depth2 = sparse_bilateral_filtering(depth2, num_iter=1)  # fixme

        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]
        camera1 = cameras[timestamp1]
        camera2 = cameras[timestamp2]
        intrinsic1 = unnormalize_intrinsics(camera1.intrinsics, h1, w1)
        intrinsic2 = unnormalize_intrinsics(camera2.intrinsics, h2, w2)

        # resize image.
        ds_factor = min(self.ds_factor, 640. / max(h1, w1))
        w_out = 640 if w1 > h1 else 360
        h_out = ((w_out / w1) * h1) // 16 * 16
        image1, intrinsic1 = resize_img_intrinsic(image1, intrinsic1, w_out, h_out)
        image2, intrinsic2 = resize_img_intrinsic(image2, intrinsic2, w_out, h_out)
        image1 = np.clip(image1, a_min=0., a_max=1.)
        image2 = np.clip(image2, a_min=0., a_max=1.)

        depth1 = resize_img(depth1, ds_factor, w_out, h_out)
        depth2 = resize_img(depth2, ds_factor, w_out, h_out)

        relative_pose = camera2.w2c_mat.dot(camera1.c2w_mat)

        return {
            'src_img1': self.to_tensor(image1).float(),
            'src_img2': self.to_tensor(image2).float(),
            'src_depth1': self.to_tensor(depth1).float(),
            'src_depth2': self.to_tensor(depth2).float(),
            'intrinsic1': torch.from_numpy(intrinsic1).float(),
            'intrinsic2': torch.from_numpy(intrinsic2).float(),
            'tgt_intrinsic': torch.from_numpy(intrinsic2).float(),
            'pose': torch.from_numpy(relative_pose).float(),
            'tgt_pose': torch.from_numpy(relative_pose).float(),
            'scale_shift1': torch.from_numpy(scale_shift1).float(),
            'scale_shift2': torch.from_numpy(scale_shift2).float(),
            'src_rgb_file1': img_file1,
            'src_rgb_file2': img_file2,
            'tgt_rgb_file': img_file2,
            'scene_dir': camera_file,
            'multi_view': True
        }