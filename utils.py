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
import numpy as np
import torch
from datetime import datetime
import shutil

TINY_NUMBER = 1e-6      # float32 only has 7 decimal digits precision


def de_parallel(model):
    return model.module if hasattr(model, 'module') else model


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def dict_to_device(dict_):
    for k in dict_.keys():
        if type(dict_[k]) == torch.Tensor:
            dict_[k] = dict_[k].cuda()

    return dict_


def save_current_code(outdir):
    now = datetime.now()  # current date and time
    date_time = now.strftime("%m_%d-%H:%M:%S")
    src_dir = '.'
    code_out_dir = os.path.join(outdir, 'code')
    os.makedirs(code_out_dir, exist_ok=True)
    dst_dir = os.path.join(code_out_dir, '{}'.format(date_time))
    shutil.copytree(src_dir, dst_dir,
                    ignore=shutil.ignore_patterns('pretrained*', '*logs*', 'out*', '*.png', '*.mp4', 'eval*',
                                                  '*__pycache__*', '*.git*', '*.idea*', '*.zip', '*.jpg'))


def nan_to_num(input, nan=0.0, posinf=None, neginf=None, *, out=None):  # pylint: disable=redefined-builtin
    # assert isinstance(input, torch.Tensor)
    if posinf is None:
        posinf = torch.finfo(input.dtype).max
    if neginf is None:
        neginf = torch.finfo(input.dtype).min
    assert nan == 0
    return torch.clamp(input.unsqueeze(0).nansum(0), min=neginf, max=posinf, out=out)


def img2mse(x, y, mask=None):
    '''
    :param x: img 1, [(...), 3]
    :param y: img 2, [(...), 3]
    :param mask: optional, [(...)]
    :return: mse score
    '''
    if mask is None:
        return torch.mean((x - y) * (x - y))
    else:
        return torch.sum((x - y) * (x - y) * mask.unsqueeze(-1)) / (torch.sum(mask) * x.shape[-1] + TINY_NUMBER)


mse2psnr = lambda x: -10. * np.log(x+TINY_NUMBER) / np.log(10.)


def img2psnr(x, y, mask=None):
    return mse2psnr(img2mse(x, y, mask).item())

