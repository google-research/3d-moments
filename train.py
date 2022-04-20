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


import config
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
import time
import torch.utils.data.distributed
import torch.distributed as dist
from tensorboardX import SummaryWriter

from data_loaders import dataset_dict
from data_loaders.create_training_dataset import create_training_dataset
from utils import *
from model import SpaceTimeModel
from core.utils import *
from criterion import Criterion
from core.scene_flow import SceneFlowEstimator
from core.renderer import ImgRenderer
from core.inpainter import Inpainter


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def train(args):
    device = "cuda:{}".format(args.local_rank)
    out_folder = os.path.join(args.rootdir, 'out', args.expname)
    print('outputs will be saved to {}'.format(out_folder))
    os.makedirs(out_folder, exist_ok=True)
    if args.local_rank == 0:
        save_current_code(out_folder)

    # save the args and config files
    f = os.path.join(out_folder, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))

    if args.config is not None:
        f = os.path.join(out_folder, 'config.txt')
        if not os.path.isfile(f):
            shutil.copy(args.config, f)

    # create training dataset
    train_dataset, train_sampler = create_training_dataset(args)

    assert args.batch_size == 1, "only support batch size == 1"
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               worker_init_fn=lambda _: np.random.seed(),
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               shuffle=True if train_sampler is None else False)

    # create validation dataset
    val_dataset = dataset_dict[args.eval_dataset](args, 'val')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1)
    val_loader_iterator = iter(cycle(val_loader))

    model = SpaceTimeModel(args)
    scene_flow_estimator = SceneFlowEstimator(args, model.raft)
    inpainter = Inpainter(args, device=device)
    renderer = ImgRenderer(args, model, scene_flow_estimator, inpainter, device)

    tb_dir = os.path.join(args.rootdir, 'logs/', args.expname)
    if args.local_rank == 0:
        writer = SummaryWriter(tb_dir)
        print('saving tensorboard files to {}'.format(tb_dir))
    scalars_to_log = {}

    global_step = model.start_step + 1
    epoch = 0
    criterion = Criterion(args)

    while global_step < model.start_step + args.n_iters + 1:
        np.random.seed()

        if args.distributed:
            train_sampler.set_epoch(epoch)

        for data in train_loader:

            if (data['src_depth1'].min() <= 1e-2) or (data['src_depth2'].min() <= 1e-2):
                continue

            start = time.time()
            res_dict = renderer.get_prediction(data)

            pred_img = res_dict['pred_img']
            gt_img = res_dict['gt_img']
            direct_rgb_out = res_dict['direct_rgb_out']
            src_img1 = res_dict['src_img1']
            src_img2 = res_dict['src_img2']
            mask = res_dict['mask']

            if res_dict['skip']:
                continue

            ### loss term
            model.optimizer.zero_grad()
            loss, scalars_to_log = criterion(pred_img, gt_img, mask, data['multi_view'][0],
                                             res_dict, scalars_to_log, global_step)
            loss.backward()

            for param in model.learnable_params:
                if param.grad is not None:
                    nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)

            model.optimizer.step()
            model.scheduler.step()
            duration = time.time() - start

            # log
            scalars_to_log['loss'] = loss.item()
            scalars_to_log['psnr'] = img2psnr(pred_img, gt_img)
            scalars_to_log['psnr_direct'] = img2psnr(direct_rgb_out, gt_img)
            scalars_to_log['multi_view'] = data['multi_view'][0]
            scalars_to_log['lr'] = model.scheduler.get_last_lr()[0]
            scalars_to_log['time'] = duration
            # Rest is logging
            if args.local_rank == 0:
                if global_step % args.i_print == 0 or global_step < 10:
                    logstr = '{} Epoch: {}  step: {} '.format(args.expname, epoch, global_step)
                    for k in scalars_to_log.keys():
                        logstr += ' {}: {:.6f}'.format(k, scalars_to_log[k])
                        writer.add_scalar(k, scalars_to_log[k], global_step)
                    print(logstr)

                if global_step % args.i_weights == 0:
                    print('Saving checkpoints at {} to {}...'.format(global_step, out_folder))
                    fpath = os.path.join(out_folder, 'model_{:06d}.pth'.format(global_step))
                    model.save_model(fpath)

                if global_step % args.i_img == 0:
                    # only the first example in the batch
                    writer.add_image('train/gt_pred_img',
                                     torch.cat([gt_img[0], direct_rgb_out[0], pred_img[0]], dim=2),
                                     global_step, dataformats='CHW')

                    # ref and src images
                    writer.add_image('train/src_imgs',
                                     torch.cat([src_img1[0], src_img2[0]], dim=2),
                                     global_step, dataformats='CHW')

                    val_data = next(val_loader_iterator)
                    torch.cuda.empty_cache()
                    model.switch_to_eval()
                    with torch.no_grad():
                        val_res_dict = renderer.get_prediction(val_data)
                        val_pred_img = val_res_dict['pred_img']
                        val_gt_img = val_res_dict['gt_img']
                        val_src_img1 = val_res_dict['src_img1']
                        val_src_img2 = val_res_dict['src_img2']
                        val_direct_rgb_out = val_res_dict['direct_rgb_out']
                    model.switch_to_train()

                    writer.add_image('val/gt_pred_img',
                                     torch.cat([val_gt_img[0], val_direct_rgb_out[0], val_pred_img[0]], dim=2),
                                     global_step, dataformats='CHW')

                    # ref and src images
                    writer.add_image('val/src_imgs',
                                     torch.cat([val_src_img1[0], val_src_img2[0]], dim=2),
                                     global_step, dataformats='CHW')

            global_step += 1
            if global_step > model.start_step + args.n_iters + 1:
                break

        epoch += 1


if __name__ == '__main__':
    args = config.config_parser()

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    train(args)
