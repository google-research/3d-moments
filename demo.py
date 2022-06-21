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

import glob
import time
import imageio
import cv2

import config
import torchvision
import torch.utils.data.distributed
from tqdm import tqdm
from model import get_raft_model
from third_party.RAFT.core.utils.utils import InputPadder
from third_party.DPT.run_monodepth import run_dpt
from utils import *
from model import SpaceTimeModel
from core.utils import *
from core.scene_flow import SceneFlowEstimator
from core.renderer import ImgRenderer
from core.inpainter import Inpainter
from data_loaders.data_utils import resize_img
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def process_boundary_mask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=5)
    dilation = cv2.dilate(closing, kernel, iterations=1, borderType=cv2.BORDER_CONSTANT, borderValue=0.)
    return dilation


def compute_optical_flow(args, img1, img2, return_np_array=False):
    raft_model = get_raft_model(args)
    with torch.no_grad():
        img1 = torch.from_numpy(img1).float().to("cuda:{}".format(args.local_rank)).permute(2, 0, 1)[None, ...]
        img2 = torch.from_numpy(img2).float().to("cuda:{}".format(args.local_rank)).permute(2, 0, 1)[None, ...]
        padder = InputPadder(img1.shape)
        image1, image2 = padder.pad(img1, img2)
        flow_low, flow_up = raft_model.module(image1, image2, iters=20, test_mode=True, padder=padder)

    del raft_model
    torch.cuda.empty_cache()
    if return_np_array:
        return flow_up.cpu().numpy().transpose(0, 2, 3, 1)
    return flow_up.permute(0, 2, 3, 1).detach()  # [B, h, w, 2]


# homography alignment
def homography_warp_pairs(args):
    input_dir = args.input_dir
    print('processing input folder {}...'.format(input_dir))
    img_files = sorted(glob.glob(os.path.join(input_dir, '*.png'))) + \
                sorted(glob.glob(os.path.join(input_dir, '*.jpg')))
    assert len(img_files) == 2, 'input folder must contain 2 images, found {} images instead'.format(len(img_files))
    warped_out_dir = os.path.join(input_dir, 'warped')
    os.makedirs(warped_out_dir, exist_ok=True)
    mask_out_dir = os.path.join(warped_out_dir, 'mask')
    os.makedirs(mask_out_dir, exist_ok=True)
    dpt_out_dir = os.path.join(input_dir, 'dpt_depth')
    os.makedirs(dpt_out_dir, exist_ok=True)
    warped_dpt_out_dir = os.path.join(warped_out_dir, 'dpt_depth')
    os.makedirs(warped_dpt_out_dir, exist_ok=True)

    dpt_model_path = 'third_party/DPT/weights/dpt_hybrid-midas-501f0c75.pt'
    run_dpt(input_path=input_dir, output_path=dpt_out_dir, model_path=dpt_model_path, optimize=False)
    disp_files = sorted(glob.glob(os.path.join(dpt_out_dir, '*.png')))

    img1 = imageio.imread(img_files[0])
    img2 = imageio.imread(img_files[1])
    disp1 = imageio.imread(disp_files[0])
    disp2 = imageio.imread(disp_files[1])

    img_h = img1.shape[0]
    img_w = img1.shape[1]
    x = np.arange(img_h)
    y = np.arange(img_w)
    coords = np.stack(np.meshgrid(y, x), -1)

    print('=========================aligning the two input images via a homography...=========================')
    flow12 = compute_optical_flow(args, img1, img2, return_np_array=True)[0]
    flow12_norm = np.linalg.norm(flow12, axis=-1)
    mask_valid = (flow12_norm < np.inf) * (disp1 > 0)
    mask_small_flow = flow12_norm < np.percentile(flow12_norm[mask_valid], 75)
    mask_valid *= mask_small_flow

    coords1 = coords + flow12
    pt1 = coords[mask_valid][::10]
    pt2 = coords1[mask_valid][::10]

    H, mask = cv2.findHomography(pt2, pt1, method=cv2.RANSAC, ransacReprojThreshold=1)
    np.savetxt(os.path.join(input_dir, 'H.txt'), H)
    img2_warped = cv2.warpPerspective(img2, H, (img_w, img_h), flags=cv2.INTER_LINEAR)
    imageio.imwrite(os.path.join(warped_out_dir, os.path.basename(img_files[0])), img1)
    imageio.imwrite(os.path.join(warped_out_dir, os.path.basename(img_files[1])), img2_warped)
    print('finished')

    disp2_warped = cv2.warpPerspective(disp2, H, (img_w, img_h), flags=cv2.INTER_LINEAR)
    scale21 = np.mean(mask_valid * (disp2_warped / np.clip(disp1, a_min=1e-6, a_max=np.inf))) / np.mean(mask_valid)
    if scale21 < 1:
        disp1 = (disp1 * scale21).astype(np.uint16)
    else:
        disp2_warped = (disp2_warped / scale21).astype(np.uint16)

    imageio.imwrite(os.path.join(warped_dpt_out_dir, os.path.basename(disp_files[0])), disp1)
    imageio.imwrite(os.path.join(warped_dpt_out_dir, os.path.basename(disp_files[1])), disp2_warped)

    # generate mask
    mask_save_dir = os.path.join(warped_out_dir, 'mask')
    os.makedirs(mask_save_dir, exist_ok=True)
    mask = 255 * np.ones((img_h, img_w), dtype=np.uint8)
    mask_warped = cv2.warpPerspective(mask, H, (img_w, img_h))
    imageio.imwrite(os.path.join(mask_save_dir, '0.png'), mask)
    imageio.imwrite(os.path.join(mask_save_dir, '1.png'), mask_warped)


def get_input_data(args, ds_factor=1):
    to_tensor = torchvision.transforms.ToTensor()
    input_dir = os.path.join(args.input_dir, 'warped')
    img_files = sorted(glob.glob(os.path.join(input_dir, '*.png'))) + \
                sorted(glob.glob(os.path.join(input_dir, '*.jpg')))
    img_file1, img_file2 = img_files
    src_img1 = imageio.imread(img_file1) / 255.
    src_img2 = imageio.imread(img_file2) / 255.
    src_img1 = resize_img(src_img1, ds_factor)
    src_img2 = resize_img(src_img2, ds_factor)

    h1, w1 = src_img1.shape[:2]
    h2, w2 = src_img2.shape[:2]

    src_disp1 = imageio.imread(os.path.join(input_dir, 'dpt_depth', os.path.basename(img_file1))) / 65535.
    src_disp2 = imageio.imread(os.path.join(input_dir, 'dpt_depth', os.path.basename(img_file2))) / 65535.

    src_disp1 = remove_noise_in_dpt_disparity(src_disp1)
    src_disp2 = remove_noise_in_dpt_disparity(src_disp2)

    src_depth1 = 1. / np.maximum(src_disp1, 1e-2)
    src_depth2 = 1. / np.maximum(src_disp2, 1e-2)

    src_depth1 = resize_img(src_depth1, ds_factor)
    src_depth2 = resize_img(src_depth2, ds_factor)

    intrinsic1 = np.array([[max(h1, w1), 0, w1 // 2],
                           [0, max(h1, w1), h1 // 2],
                           [0, 0, 1]])

    intrinsic2 = np.array([[max(h2, w2), 0, w2 // 2],
                           [0, max(h2, w2), h2 // 2],
                           [0, 0, 1]])

    pose = np.eye(4)
    return {
        'src_img1': to_tensor(src_img1).float()[None],
        'src_img2': to_tensor(src_img2).float()[None],
        'src_depth1': to_tensor(src_depth1).float()[None],
        'src_depth2': to_tensor(src_depth2).float()[None],
        'intrinsic1': torch.from_numpy(intrinsic1).float()[None],
        'intrinsic2': torch.from_numpy(intrinsic2).float()[None],
        'tgt_intrinsic': torch.from_numpy(intrinsic2).float()[None],
        'pose': torch.from_numpy(pose).float()[None],
        'scale_shift1': torch.tensor([1., 0.]).float()[None],
        'scale_shift2': torch.tensor([1., 0.]).float()[None],
        'src_rgb_file1': [img_file1],
        'src_rgb_file2': [img_file2],
        'multi_view': [False]
    }


def render(args):
    device = "cuda:{}".format(args.local_rank)
    homography_warp_pairs(args)

    print('=========================run 3D Moments...=========================')

    data = get_input_data(args)
    rgb_file1 = data['src_rgb_file1'][0]
    rgb_file2 = data['src_rgb_file2'][0]
    frame_id1 = os.path.basename(rgb_file1).split('.')[0]
    frame_id2 = os.path.basename(rgb_file2).split('.')[0]
    scene_id = rgb_file1.split('/')[-3]

    video_out_folder = os.path.join(args.input_dir, 'out')
    os.makedirs(video_out_folder, exist_ok=True)

    model = SpaceTimeModel(args)
    if model.start_step == 0:
        raise Exception('no pretrained model found! please check the model path.')

    scene_flow_estimator = SceneFlowEstimator(args, model.raft)
    inpainter = Inpainter(args)
    renderer = ImgRenderer(args, model, scene_flow_estimator, inpainter, device)

    model.switch_to_eval()
    with torch.no_grad():
        renderer.process_data(data)

        pts1, pts2, rgb1, rgb2, feat1, feat2, mask, side_ids, optical_flow = \
            renderer.render_rgbda_layers_with_scene_flow(return_pts=True)

        num_frames = [60, 60, 60, 90]
        video_paths = ['up-down', 'zoom-in', 'side', 'circle']
        Ts = [
            define_camera_path(num_frames[0], 0., -0.08, 0., path_type='double-straight-line', return_t_only=True),
            define_camera_path(num_frames[1], 0., 0., -0.24, path_type='straight-line', return_t_only=True),
            define_camera_path(num_frames[2], -0.09, 0, -0, path_type='double-straight-line', return_t_only=True),
            define_camera_path(num_frames[3], -0.04, -0.04, -0.09, path_type='circle', return_t_only=True),
        ]
        crop = 32

        for j, T in enumerate(Ts):
            T = torch.from_numpy(T).float().to(renderer.device)
            time_steps = np.linspace(0, 1, num_frames[j])
            frames = []
            for i, t_step in tqdm(enumerate(time_steps), total=len(time_steps),
                                  desc='generating video of {} camera trajectory'.format(video_paths[j])):
                pred_img, _, meta = renderer.render_pcd(pts1, pts2, rgb1, rgb2,
                                                        feat1, feat2, mask, side_ids,
                                                        t=T[i], time=t_step)
                frame = (255. * pred_img.detach().cpu().squeeze().permute(1, 2, 0).numpy()).astype(np.uint8)
                # mask out fuzzy image boundaries due to no outpainting
                img_boundary_mask = (meta['acc'] > 0.5).detach().cpu().squeeze().numpy().astype(np.uint8)
                img_boundary_mask_cleaned = process_boundary_mask(img_boundary_mask)
                frame = frame * img_boundary_mask_cleaned[..., None]
                frame = frame[crop:-crop, crop:-crop]
                frames.append(frame)

            video_out_file = os.path.join(video_out_folder, '{}_{}-{}-{}.mp4'.format(
                video_paths[j], scene_id, frame_id1, frame_id2))
            imageio.mimwrite(video_out_file, frames, fps=25, quality=8)

        print('space-time videos have been saved in {}.'.format(video_out_folder))


if __name__ == '__main__':
    args = config.config_parser()
    render(args)
