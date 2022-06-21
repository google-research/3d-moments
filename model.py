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
import torch
from networks.resunet import ResUNet
from networks.img_decoder import ImgDecoder
from third_party.RAFT.core.raft import RAFT
from utils import de_parallel


class Namespace:

    def __init__(self, **kwargs):
        for name in kwargs:
            setattr(self, name, kwargs[name])

    def __eq__(self, other):
        if not isinstance(other, Namespace):
            return NotImplemented
        return vars(self) == vars(other)

    def __contains__(self, key):
        return key in self.__dict__


def get_raft_model(args):
    flow_args = Namespace()
    setattr(flow_args, 'model', 'third_party/RAFT/models/raft-things.pth')
    setattr(flow_args, 'small', False)
    setattr(flow_args, 'mixed_precision', False)
    setattr(flow_args, 'alternate_corr', False)

    device = "cuda:{}".format(args.local_rank)
    if args.distributed:
        model = RAFT(flow_args).to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
        )
    else:
        model = torch.nn.DataParallel(RAFT(flow_args)).to(device)

    model.load_state_dict(torch.load(flow_args.model, map_location='cuda:{}'.format(args.local_rank)))
    return model


########################################################################################################################
# creation/saving/loading of the model
########################################################################################################################


class SpaceTimeModel(object):
    def __init__(self, args):
        self.args = args
        load_opt = not args.no_load_opt
        load_scheduler = not args.no_load_scheduler
        device = torch.device('cuda:{}'.format(args.local_rank))

        # initialize feature extraction network
        feat_in_ch = 4
        if args.use_inpainting_mask_for_feature:
            feat_in_ch += 1
        if args.use_depth_for_feature:
            feat_in_ch += 1
        self.feature_net = ResUNet(args, in_ch=feat_in_ch, out_ch=args.feature_dim).to(device)
        # initialize decoder
        decoder_in_ch = args.feature_dim + 4
        decoder_out_ch = 3

        if args.use_depth_for_decoding:
            decoder_in_ch += 1
        if args.use_mask_for_decoding:
            decoder_in_ch += 1

        self.img_decoder = ImgDecoder(args, in_ch=decoder_in_ch, out_ch=decoder_out_ch).to(device)
        self.raft = get_raft_model(args)

        learnable_params = list(self.feature_net.parameters())
        learnable_params += list(self.img_decoder.parameters())

        self.learnable_params = learnable_params
        if args.train_raft:
            self.optimizer = torch.optim.Adam([
                {'params': learnable_params},
                {'params': filter(lambda p: p.requires_grad, self.raft.parameters()), 'lr': self.args.lr_raft}],
                lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.999))
        else:
            self.raft.eval()
            self.optimizer = torch.optim.Adam(learnable_params, lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.999))

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=args.lrate_decay_steps,
                                                         gamma=args.lrate_decay_factor)

        out_folder = os.path.join(args.rootdir, 'out', args.expname)
        self.start_step = self.load_from_ckpt(out_folder,
                                              load_opt=load_opt,
                                              load_scheduler=load_scheduler)

        if args.distributed:
            self.feature_net = torch.nn.parallel.DistributedDataParallel(
                self.feature_net,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
            )

            self.img_decoder = torch.nn.parallel.DistributedDataParallel(
                self.img_decoder,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
            )

    def switch_to_eval(self):
        self.feature_net.eval()
        self.img_decoder.eval()
        self.raft.eval()

    def switch_to_train(self):
        self.feature_net.train()
        self.img_decoder.train()
        if self.args.train_raft:
            self.raft.train()

    def save_model(self, filename):
        to_save = {'optimizer': self.optimizer.state_dict(),
                   'scheduler': self.scheduler.state_dict(),
                   'feature_net': de_parallel(self.feature_net).state_dict(),
                   'img_decoder': de_parallel(self.img_decoder).state_dict(),
                   'raft': self.raft.state_dict()
                   }
        torch.save(to_save, filename)

    def load_model(self, filename, load_opt=True, load_scheduler=True):
        if self.args.distributed:
            to_load = torch.load(filename, map_location='cuda:{}'.format(self.args.local_rank))
        else:
            to_load = torch.load(filename)

        if load_opt:
            self.optimizer.load_state_dict(to_load['optimizer'])
        if load_scheduler:
            self.scheduler.load_state_dict(to_load['scheduler'])

        self.feature_net.load_state_dict(to_load['feature_net'])
        self.img_decoder.load_state_dict(to_load['img_decoder'])
        if 'raft' in to_load.keys():
            self.raft.load_state_dict(to_load['raft'])

    def load_from_ckpt(self, out_folder,
                       load_opt=True,
                       load_scheduler=True,
                       force_latest_ckpt=False):
        '''
        load model from existing checkpoints and return the current step
        :param out_folder: the directory that stores ckpts
        :return: the current starting step
        '''

        # all existing ckpts
        ckpts = []
        if os.path.exists(out_folder):
            ckpts = [os.path.join(out_folder, f)
                     for f in sorted(os.listdir(out_folder)) if f.endswith('.pth')]

        if self.args.ckpt_path is not None and not force_latest_ckpt:
            if os.path.isfile(self.args.ckpt_path):  # load the specified ckpt
                ckpts = [self.args.ckpt_path]

        if len(ckpts) > 0 and not self.args.no_reload:
            fpath = ckpts[-1]
            self.load_model(fpath, load_opt, load_scheduler)
            step = int(fpath[-10:-4])
            print('Reloading from {}, starting at step={}'.format(fpath, step))
        else:
            print('No ckpts found, training from scratch...')
            step = 0

        return step


