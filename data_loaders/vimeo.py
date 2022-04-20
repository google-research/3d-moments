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
import glob
from torch.utils.data import Dataset
from data_loaders.data_utils import get_src_tgt_ids, resize_img
import torchvision
from core.utils import remove_noise_in_dpt_disparity


def get_black_boundary_size(img):
    h, w = img.shape[:2]
    mean_img = np.mean(img, axis=-1)
    mask_mean_x_axis = mean_img.mean(axis=0)
    x_valid = np.nonzero(mask_mean_x_axis > 1e-3)
    if len(x_valid[0]) == 0:
        left, right = 0, w
    else:
        left, right = x_valid[0][0], x_valid[0][-1]+1
    mask_mean_y_axis = mean_img.mean(axis=1)
    y_valid = np.nonzero(mask_mean_y_axis > 1e-3)
    if len(y_valid[0]) == 0:
        top, bottom = 0, h
    else:
        top, bottom = y_valid[0][0], y_valid[0][-1]+1
    assert 0 <= top <= h and 0 <= bottom <= h and 0 <= left <= w and 0 <= right <= w
    top = top + (16 - top % 16) if top % 16 != 0 else top
    left = left + (16 - left % 16) if left % 16 != 0 else left
    bottom = bottom - bottom % 16 if bottom % 16 != 0 else bottom
    right = right - right % 16 if right % 16 != 0 else right
    if bottom - top < 128:
        top = 0
        bottom = h
    if right - left < 128:
        left = 0
        right = w
    return top, bottom, left, right


class VimeoDataset(Dataset):
    def __init__(self, args, subset, **kwargs):
        base_dir = 'data/vimeo/sequences/'
        scene_dirs = sorted(glob.glob(os.path.join(base_dir, '*/*')))
        if subset == 'train':
            self.scene_dirs = scene_dirs[:-100]
        else:
            self.scene_dirs = scene_dirs[-100:]
        self.to_tensor = torchvision.transforms.ToTensor()
        self.ds_factor = 1

    def __len__(self):
        return len(self.scene_dirs)

    def __getitem__(self, idx):
        scene_dir = self.scene_dirs[idx]
        img_dir = scene_dir
        depth_dir = os.path.join(scene_dir, 'dpt_depth')
        img_files = sorted(glob.glob(os.path.join(img_dir, '*.png')))
        dpt_files = sorted(glob.glob(os.path.join(depth_dir, '*.png')))
        assert len(img_files) == len(dpt_files), print(scene_dir)
        num_frames = len(img_files)
        src_id1, src_id2, tgt_id = get_src_tgt_ids(num_frames, max_interval=3)
        if np.random.choice([0, 1], p=[0.996, 0.004]):
            src_id = np.random.choice([src_id1, src_id2])
            tgt_id = src_id
        time = np.clip((tgt_id - src_id1 + np.random.rand() - 0.5) / (src_id2 - src_id1), a_min=0., a_max=1.)

        src_img1 = imageio.imread(img_files[src_id1]) / 255.
        src_img2 = imageio.imread(img_files[src_id2]) / 255.
        tgt_img = imageio.imread(img_files[tgt_id]) / 255.
        t, b, l, r = get_black_boundary_size(src_img1)

        src_img1 = resize_img(src_img1, self.ds_factor)
        src_img2 = resize_img(src_img2, self.ds_factor)
        tgt_img = resize_img(tgt_img, self.ds_factor)

        src_disp1 = imageio.imread(dpt_files[src_id1]) / 65535.
        src_disp2 = imageio.imread(dpt_files[src_id2]) / 65535.
        src_disp1 = remove_noise_in_dpt_disparity(src_disp1)
        src_disp2 = remove_noise_in_dpt_disparity(src_disp2)
        src_depth1 = 1. / np.maximum(src_disp1, 1e-2)
        src_depth2 = 1. / np.maximum(src_disp2, 1e-2)
        # src_depth1 = sparse_bilateral_filtering(src_depth1, num_iter=1)  # fixme
        # src_depth2 = sparse_bilateral_filtering(src_depth2, num_iter=1)

        src_depth1 = resize_img(src_depth1, self.ds_factor)
        src_depth2 = resize_img(src_depth2, self.ds_factor)

        src_img1 = src_img1[t:b, l:r]
        src_img2 = src_img2[t:b, l:r]
        tgt_img = tgt_img[t:b, l:r]
        src_depth1 = src_depth1[t:b, l:r]
        src_depth2 = src_depth2[t:b, l:r]

        h1, w1 = src_img1.shape[:2]
        h2, w2 = src_img2.shape[:2]
        ht, wt = tgt_img.shape[:2]

        intrinsic1 = np.array([[max(h1, w1), 0, w1 // 2],
                               [0, max(h1, w1), h1 // 2],
                               [0, 0, 1]])

        intrinsic2 = np.array([[max(h2, w2), 0, w2 // 2],
                               [0, max(h2, w2), h2 // 2],
                               [0, 0, 1]])

        tgt_intrinsic = np.array([[max(ht, wt), 0, wt // 2],
                                  [0, max(ht, wt), ht // 2],
                                  [0, 0, 1]])

        relative_pose = np.eye(4)
        tgt_pose = np.eye(4)

        return {
            'src_img1': self.to_tensor(src_img1).float(),
            'src_img2': self.to_tensor(src_img2).float(),
            'src_depth1': self.to_tensor(src_depth1).float(),
            'src_depth2': self.to_tensor(src_depth2).float(),
            'tgt_img': self.to_tensor(tgt_img).float(),
            'intrinsic1': torch.from_numpy(intrinsic1).float(),
            'intrinsic2': torch.from_numpy(intrinsic2).float(),
            'tgt_intrinsic': torch.from_numpy(tgt_intrinsic).float(),
            'pose': torch.from_numpy(relative_pose).float(),
            'tgt_pose': torch.from_numpy(tgt_pose).float(),
            'scale_shift1': torch.tensor([1., 0.]).float(),
            'scale_shift2': torch.tensor([1., 0.]).float(),
            'time': time,
            'src_rgb_file1': img_files[src_id1],
            'src_rgb_file2': img_files[src_id2],
            'tgt_rgb_file': img_files[tgt_id],
            'scene_dir': scene_dir,
            'multi_view': False
        }


if __name__ == '__main__':
    dataset = VimeoDataset()
    for data in dataset:
        continue
