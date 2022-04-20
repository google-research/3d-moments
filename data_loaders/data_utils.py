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


import cv2
import numpy as np


def resize_img_intrinsic(img, intrinsic, w_out, h_out):
    h, w = img.shape[:2]
    if w_out > w:
        interpolation_method = cv2.INTER_LINEAR
    else:
        interpolation_method = cv2.INTER_AREA
    img = cv2.resize(img, (int(w_out), int(h_out)), interpolation=interpolation_method)
    intrinsic[0] *= 1. * w_out / w
    intrinsic[1] *= 1. * h_out / h
    return img, intrinsic


def resize_img(img, ds_factor, w_out=None, h_out=None):
    h, w = img.shape[:2]
    if w_out is None and h_out is None:
        if ds_factor == 1:
            return img
        if ds_factor > 1:
            interpolation_method = cv2.INTER_LINEAR
        else:
            interpolation_method = cv2.INTER_AREA
        img = cv2.resize(img, (int(w*ds_factor), int(h*ds_factor)), interpolation=interpolation_method)
    else:
        if w_out > w:
            interpolation_method = cv2.INTER_LINEAR
        else:
            interpolation_method = cv2.INTER_AREA
        img = cv2.resize(img, (int(w_out), int(h_out)), interpolation=interpolation_method)
    return img


def get_src_tgt_ids(num_frames, max_interval=1):
    assert num_frames > max_interval + 1
    src_id1 = np.random.choice(num_frames-max_interval-1)
    interval = np.random.randint(low=0, high=max_interval) + 1
    src_id2 = src_id1 + interval + 1
    tgt_id = np.random.randint(src_id1 + 1, src_id2)
    return src_id1, src_id2, tgt_id


def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


class Camera(object):
    def __init__(self, entry):
        fx, fy, cx, cy = entry[1:5]
        self.intrinsics = np.array([[fx, 0, cx, 0],
                                    [0, fy, cy, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)


def unnormalize_intrinsics(intrinsics, h, w):
    intrinsics[0] *= w
    intrinsics[1] *= h
    return intrinsics


def parse_pose_file(file):
    f = open(file, 'r')
    cam_params = {}
    for i, line in enumerate(f):
        if i == 0:
            video_id = line.replace('https://www.youtube.com/watch?v=', '')[:-1]
            continue
        entry = [float(x) for x in line.split()]
        id = int(entry[0])
        cam_params[id] = Camera(entry)
    return video_id, cam_params


def crop_img(img, factor=16):
    h, w = img.shape[:2]
    ho = h // factor * factor
    wo = w // factor * factor
    img = img[:ho, :wo]
    return img
