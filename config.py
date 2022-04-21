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

import configargparse


def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, help='config file path')
    # general
    parser.add_argument('--rootdir', type=str, default='./',
                        help='the path to the project root directory.')
    parser.add_argument("--expname", type=str, default='exp', help='experiment name')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--distributed', action='store_true', help='if use distributed training')
    parser.add_argument("--local_rank", type=int, default=0, help='rank for distributed training')
    parser.add_argument("--eval_mode", action='store_true', help='if in eval mode')

    ########## dataset options ##########
    # train and eval dataset
    parser.add_argument("--train_dataset", type=str, default='vimeo',
                        help='the training dataset')
    parser.add_argument("--dataset_weights", nargs='+', type=float, default=[],
                        help='the weights for training datasets, used when multiple datasets are used.')
    parser.add_argument('--eval_dataset', type=str, default='vimeo', help='the dataset to evaluate')
    parser.add_argument("--batch_size", type=int, default=1, help='batch size, currently only support 1')

    ########## network architecture ##########
    parser.add_argument("--feature_dim", type=int, default=32, help='the dimension of the extracted features')

    ########## training options ##########
    parser.add_argument("--use_inpainting_mask_for_feature", action='store_true')
    parser.add_argument("--inpainting", action='store_true', help='if do inpainting')
    parser.add_argument("--train_raft", action='store_true', help='if train raft')
    parser.add_argument('--boundary_crop_ratio', type=float, default=0, help='crop the image before computing loss')
    parser.add_argument("--vary_pts_radius", action='store_true', help='if vary point radius as augmentation')
    parser.add_argument("--adaptive_pts_radius", action='store_true', help='if use adaptive point radius')
    parser.add_argument("--use_mask_for_decoding", action='store_true', help='if use mask for decoding')

    ########## rendering/evaluation ##########
    parser.add_argument("--use_depth_for_feature", action='store_true',
                        help='if use depth map when extracting features')
    parser.add_argument("--use_depth_for_decoding", action='store_true',
                        help='if use depth map when decoding')
    parser.add_argument("--point_radius", type=float, default=1.5,
                        help='point radius for rasterization')
    parser.add_argument("--input_dir", type=str, default='', help='input folder that contains a pair of images')
    parser.add_argument("--visualize_rgbda_layers", action='store_true',
                        help="if visualize rgbda layers, save in out dir")

    ########### iterations & learning rate options & loss ##########
    parser.add_argument("--n_iters", type=int, default=250000, help='num of iterations')
    parser.add_argument("--lr", type=float, default=3e-4, help='learning rate for feature extractor')
    parser.add_argument("--lr_raft", type=float, default=5e-6, help='learning rate for raft')
    parser.add_argument("--lrate_decay_factor", type=float, default=0.5,
                        help='decay learning rate by a factor every specified number of steps')
    parser.add_argument("--lrate_decay_steps", type=int, default=50000,
                        help='decay learning rate by a factor every specified number of steps')
    parser.add_argument('--loss_mode', type=str, default='lpips',
                        help='the loss function to use')

    ########## checkpoints ##########
    parser.add_argument("--ckpt_path", type=str, default="",
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--no_load_opt", action='store_true',
                        help='do not load optimizer when reloading')
    parser.add_argument("--no_load_scheduler", action='store_true',
                        help='do not load scheduler when reloading')

    ########## logging/saving options ##########
    parser.add_argument("--i_print", type=int, default=100, help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img", type=int, default=500, help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, help='frequency of weight ckpt saving')

    args = parser.parse_args()
    return args

