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
import torch.utils.data.distributed
from pytorch3d.structures import Pointclouds
from core.utils import *
from core.depth_layering import get_depth_bins
from core.pcd import linear_interpolation, create_pcd_renderer


class ImgRenderer():
    def __init__(self, args, model, scene_flow_estimator, inpainter, device):
        self.args = args
        self.model = model
        self.scene_flow_estimator = scene_flow_estimator
        self.inpainter = inpainter
        self.device = device

    def process_data(self, data):
        self.src_img1 = data['src_img1'].to(self.device)
        self.src_img2 = data['src_img2'].to(self.device)
        assert self.src_img1.shape == self.src_img2.shape
        self.h, self.w = self.src_img1.shape[-2:]
        self.src_depth1 = data['src_depth1'].to(self.device)
        self.src_depth2 = data['src_depth2'].to(self.device)
        self.intrinsic1 = data['intrinsic1'].to(self.device)
        self.intrinsic2 = data['intrinsic2'].to(self.device)

        self.pose = data['pose'].to(self.device)
        self.scale_shift1 = data['scale_shift1'][0]
        self.scale_shift2 = data['scale_shift2'][0]
        self.is_multi_view = data['multi_view'][0]
        self.src_rgb_file1 = data['src_rgb_file1'][0]
        self.src_rgb_file2 = data['src_rgb_file2'][0]
        if 'tgt_img' in data.keys():
            self.tgt_img = data['tgt_img'].to(self.device)
        if 'tgt_intrinsic' in data.keys():
            self.tgt_intrinsic = data['tgt_intrinsic'].to(self.device)
        if 'tgt_pose' in data.keys():
            self.tgt_pose = data['tgt_pose'].to(self.device)
        if 'time' in data.keys():
            self.time = data['time'].item()
        if 'src_mask1' in data.keys():
            self.src_mask1 = data['src_mask1'].to(self.device)
        else:
            self.src_mask1 = torch.ones_like(self.src_depth1)
        if 'src_mask2' in data.keys():
            self.src_mask2 = data['src_mask2'].to(self.device)
        else:
            self.src_mask2 = torch.ones_like(self.src_depth2)

    def feature_extraction(self, rgba_layers, mask_layers, depth_layers):
        rgba_layers_in = rgba_layers.squeeze(1)

        if self.args.use_inpainting_mask_for_feature:
            rgba_layers_in = torch.cat([rgba_layers_in, mask_layers.squeeze(1)], dim=1)

        if self.args.use_depth_for_feature:
            rgba_layers_in = torch.cat([rgba_layers_in, 1. / torch.clamp(depth_layers.squeeze(1), min=1.)], dim=1)
        featmaps = self.model.feature_net(rgba_layers_in)
        return featmaps

    def apply_scale_shift(self, depth, scale, shift):
        disp = 1. / torch.clamp(depth, min=1e-3)
        disp = scale * disp + shift
        return 1 / torch.clamp(disp, min=1e-3*scale)

    def masked_diffuse(self, x, mask, iter=10, kernel_size=35, median_blur=False):
        if median_blur:
            x = masked_median_blur(x, mask.repeat(1, x.shape[1], 1, 1), kernel_size=5)
        for _ in range(iter):
            x, mask = masked_smooth_filter(x, mask, kernel_size=kernel_size)
        return x, mask

    def compute_weight_for_two_frame_blending(self, time, disp1, disp2, alpha1, alpha2):
        alpha = 4
        weight1 = (1 - time) * torch.exp(alpha*disp1) * alpha1
        weight2 = time * torch.exp(alpha*disp2) * alpha2
        sum_weight = torch.clamp(weight1 + weight2, min=1e-6)
        out_weight1 = weight1 / sum_weight
        out_weight2 = weight2 / sum_weight
        return out_weight1, out_weight2

    def transform_all_pts(self, all_pts, pose):
        all_pts_out = []
        for pts in all_pts:
            pts_out = transform_pts_in_3D(pts, pose)
            all_pts_out.append(pts_out)
        return all_pts_out

    def render_pcd(self, pts1, pts2, rgbs1, rgbs2, feats1, feats2, mask, side_ids, R=None, t=None, time=0):

        pts = linear_interpolation(pts1, pts2, time)
        rgbs = linear_interpolation(rgbs1, rgbs2, time)
        feats = linear_interpolation(feats1, feats2, time)
        rgb_feat = torch.cat([rgbs, feats], dim=-1)

        num_sides = side_ids.max() + 1
        assert num_sides == 1 or num_sides == 2

        if R is None:
            R = torch.eye(3, device=self.device)
        if t is None:
            t = torch.zeros(3, device=self.device)

        pts_ = (R.mm(pts.T) + t.unsqueeze(-1)).T
        if self.args.adaptive_pts_radius:
            radius = self.args.point_radius / min(self.h, self.w) * 2.0 * pts[..., -1][None] / \
                     torch.clamp(pts_[..., -1][None], min=1e-6)
        else:
            radius = self.args.point_radius / min(self.h, self.w) * 2.0

        if self.args.vary_pts_radius and np.random.choice([0, 1], p=[0.6, 0.4]):
            if type(radius) == torch.Tensor:
                factor = 1 + (0.2 * (torch.rand_like(radius) - 0.5))
            else:
                factor = 1 + (0.2 * (np.random.rand() - 0.5))
            radius *= factor

        if self.args.use_mask_for_decoding:
            rgb_feat = torch.cat([rgb_feat, mask], dim=-1)

        if self.args.use_depth_for_decoding:
            disp = normalize_0_1(1. / torch.clamp(pts_[..., [-1]], min=1e-6))
            rgb_feat = torch.cat([rgb_feat, disp], dim=-1)

        global_out_list = []
        direct_color_out_list = []
        meta = {}
        for j in range(num_sides):
            mask_side = side_ids == j
            renderer = create_pcd_renderer(self.h, self.w, self.tgt_intrinsic.squeeze()[:3, :3],
                                           radius=radius[:, mask_side] if type(radius) == torch.Tensor else radius)
            all_pcd_j = Pointclouds(points=[pts_[mask_side]], features=[rgb_feat[mask_side]])
            global_out_j = renderer(all_pcd_j)
            all_colored_pcd_j = Pointclouds(points=[pts_[mask_side]], features=[rgbs[mask_side]])
            direct_rgb_out_j = renderer(all_colored_pcd_j)

            global_out_list.append(global_out_j)
            direct_color_out_list.append(direct_rgb_out_j)

        w1, w2 = self.compute_weight_for_two_frame_blending(time,
                                                            global_out_list[0][..., [-1]],
                                                            global_out_list[-1][..., [-1]],
                                                            global_out_list[0][..., [3]],
                                                            global_out_list[-1][..., [3]]
                                                            )
        direct_rgb_out = w1 * direct_color_out_list[0] + w2 * direct_color_out_list[-1]
        pred_rgb = self.model.img_decoder(global_out_list[0].permute(0, 3, 1, 2),
                                          global_out_list[-1].permute(0, 3, 1, 2),
                                          time)

        direct_rgb = direct_rgb_out[..., :3].permute(0, 3, 1, 2)
        acc = 0.5 * (global_out_list[0][..., [3]] + global_out_list[1][..., [3]]).permute(0, 3, 1, 2)
        meta['acc'] = acc
        return pred_rgb, direct_rgb, meta

    def get_reprojection_mask(self, pts, R, t):
        pts1_ = (R.mm(pts.T) + t.unsqueeze(-1)).T
        mask1 = torch.ones_like(self.src_img1[:, :1].reshape(-1, 1))
        mask_renderer = create_pcd_renderer(self.h, self.w, self.tgt_intrinsic.squeeze()[:3, :3],
                                            radius=1.0 / min(self.h, self.w) * 4.)
        mask_pcd = Pointclouds(points=[pts1_], features=[mask1])
        mask = mask_renderer(mask_pcd).permute(0, 3, 1, 2)
        mask = F.max_pool2d(mask, kernel_size=7, stride=1, padding=3)
        return mask

    def get_cropping_ids(self, mask):
        assert mask.shape[:2] == (1, 1)
        mask = mask.squeeze()
        h, w = mask.shape
        mask_mean_x_axis = mask.mean(dim=0)
        x_valid = torch.nonzero(mask_mean_x_axis > 0.5)
        bad = False
        if len(x_valid) < 0.75 * w:
            left, right = 0, w - 1   # invalid
            bad = True
        else:
            left, right = x_valid[0][0], x_valid[-1][0]
        mask_mean_y_axis = mask.mean(dim=1)
        y_valid = torch.nonzero(mask_mean_y_axis > 0.5)
        if len(y_valid) < 0.75 * h:
            top, bottom = 0, h - 1   # invalid
            bad = True
        else:
            top, bottom = y_valid[0][0], y_valid[-1][0]
        assert 0 <= top <= h - 1 and 0 <= bottom <= h - 1 and 0 <= left <= w - 1 and 0 <= right <= w - 1
        return top, bottom, left, right, bad

    def render_depth_from_mdi(self, depth_layers, alpha_layers):
        '''
        :param depth_layers: [n_layers, 1, h, w]
        :param alpha_layers: [n_layers, 1, h, w]
        :return: rendered depth [1, 1, h, w]
        '''
        num_layers = len(depth_layers)
        h, w = depth_layers.shape[-2:]
        layer_id = torch.arange(num_layers, device=self.device).float()
        layer_id_maps = layer_id[..., None, None, None, None].repeat(1, 1, 1, h, w)
        T = torch.cumprod(1. - alpha_layers, dim=0)[:-1]
        T = torch.cat([torch.ones_like(T[:1]), T], dim=0)
        weights = alpha_layers * T
        depth_map = torch.sum(weights * depth_layers, dim=0)
        depth_map = torch.clamp(depth_map, min=1.)
        layer_id_map = torch.sum(weights * layer_id_maps, dim=0)
        return depth_map, layer_id_map

    def render_rgbda_layers_from_one_view(self):
        depth_bins = get_depth_bins(depth=self.src_depth1)
        rgba_layers, depth_layers, mask_layers = \
            self.inpainter.sequential_inpainting(self.src_img1, self.src_depth1, depth_bins)
        coord1 = get_coord_grids_pt(self.h, self.w, device=self.device).float()
        src_depth1 = self.apply_scale_shift(self.src_depth1, self.scale_shift1[0], self.scale_shift1[1])
        pts1 = unproject_pts_pt(self.intrinsic1, coord1.reshape(-1, 2), src_depth1.flatten())

        featmaps = self.feature_extraction(rgba_layers, mask_layers, depth_layers)
        depth_layers = self.apply_scale_shift(depth_layers, self.scale_shift1[0], self.scale_shift1[1])
        num_layers = len(rgba_layers)
        all_pts = []
        all_rgbas = []
        all_feats = []
        all_masks = []
        for i in range(num_layers):
            alpha_i = rgba_layers[i][:, -1] > 0.5
            rgba_i = rgba_layers[i]
            mask_i = mask_layers[i]
            featmap = featmaps[i][None]
            featmap = F.interpolate(featmap, size=(self.h, self.w), mode='bilinear', align_corners=True)
            pts1_i = unproject_pts_pt(self.intrinsic1, coord1.reshape(-1, 2), depth_layers[i].flatten())
            pts1_i = pts1_i.reshape(1, self.h, self.w, 3)
            all_pts.append(pts1_i[alpha_i])
            all_rgbas.append(rgba_i.permute(0, 2, 3, 1)[alpha_i])
            all_feats.append(featmap.permute(0, 2, 3, 1)[alpha_i])
            all_masks.append(mask_i.permute(0, 2, 3, 1)[alpha_i])

        all_pts = torch.cat(all_pts)
        all_rgbas = torch.cat(all_rgbas)
        all_feats = torch.cat(all_feats)
        all_masks = torch.cat(all_masks)
        all_side_ids = torch.zeros_like(all_masks.squeeze(), dtype=torch.long)

        R = self.tgt_pose[0, :3, :3]
        t = self.tgt_pose[0, :3, 3]

        pred_img, direct_rgb_out, meta = self.render_pcd(all_pts, all_pts,
                                                         all_rgbas, all_rgbas,
                                                         all_feats, all_feats,
                                                         all_masks, all_side_ids,
                                                         R, t, 0)

        mask = self.get_reprojection_mask(pts1, R, t)
        t, b, l, r, bad = self.get_cropping_ids(mask)
        gt_img = self.src_img2
        skip = False
        if not skip and not self.args.eval_mode:
            pred_img = pred_img[:, :, t:b, l:r]
            mask = mask[:, :, t:b, l:r]
            direct_rgb_out = direct_rgb_out[:, :, t:b, l:r]
            gt_img = gt_img[:, :, t:b, l:r]
        else:
            skip = True

        res_dict = {
            'src_img1': self.src_img1,
            'src_img2': self.src_img2,
            'pred_img': pred_img,
            'gt_img': gt_img,
            'mask': mask,
            'direct_rgb_out': direct_rgb_out,
            'skip': skip
        }
        return res_dict

    def compute_scene_flow_one_side(self, coord, pose,
                                    rgb1, rgb2,
                                    rgba_layers1, rgba_layers2,
                                    featmaps1, featmaps2,
                                    pts1, pts2,
                                    depth_layers1, depth_layers2,
                                    mask_layers1, mask_layers2,
                                    flow_f, flow_b, kernel,
                                    with_inpainted=False):

        num_layers1 = len(rgba_layers1)
        pts2 = transform_pts_in_3D(pts2, pose).T.reshape(1, 3, self.h, self.w)

        mask_mutual_flow = self.scene_flow_estimator.get_mutual_matches(flow_f, flow_b, th=5, return_mask=True).float()
        mask_mutual_flow = mask_mutual_flow.unsqueeze(1)

        coord1_corsp = coord + flow_f
        coord1_corsp_normed = normalize_for_grid_sample(coord1_corsp, self.h, self.w)
        pts2_sampled = F.grid_sample(pts2, coord1_corsp_normed,  align_corners=True,
                                     mode='nearest', padding_mode="border")
        depth2_sampled = pts2_sampled[:, -1:]

        rgb2_sampled = F.grid_sample(rgb2, coord1_corsp_normed, align_corners=True, padding_mode="border")
        mask_layers2_ds = F.interpolate(mask_layers2.squeeze(1), size=featmaps2.shape[-2:], mode='area')
        featmap2 = torch.sum(featmaps2 * mask_layers2_ds, dim=0, keepdim=True)
        context2 = torch.sum(mask_layers2_ds, dim=0, keepdim=True)
        featmap2_sampled = F.grid_sample(featmap2, coord1_corsp_normed, align_corners=True, padding_mode="border")
        context2_sampled = F.grid_sample(context2, coord1_corsp_normed, align_corners=True, padding_mode="border")
        mask2_sampled = F.grid_sample(self.src_mask2, coord1_corsp_normed, align_corners=True, padding_mode="border")

        featmap2_sampled = featmap2_sampled / torch.clamp(context2_sampled, min=1e-6)
        context2_sampled = (context2_sampled > 0.5).float()
        last_pts2_i = torch.zeros_like(pts2.permute(0, 2, 3, 1))
        last_alpha_i = torch.zeros_like(rgba_layers1[0][:, -1], dtype=torch.bool)

        all_pts = []
        all_rgbas = []
        all_feats = []
        all_rgbas_end = []
        all_feats_end = []
        all_masks = []
        all_pts_end = []
        all_optical_flows = []
        for i in range(num_layers1):
            alpha_i = (rgba_layers1[i][:, -1]*self.src_mask1.squeeze(1)*mask2_sampled.squeeze(1)) > 0.5
            rgba_i = rgba_layers1[i]
            mask_i = mask_layers1[i]
            mask_no_mutual_flow = mask_i * context2_sampled
            mask_gau_i = mask_no_mutual_flow * mask_mutual_flow
            mask_no_mutual_flow = erosion(mask_no_mutual_flow, kernel)
            mask_gau_i = erosion(mask_gau_i, kernel)

            featmap1 = featmaps1[i][None]
            featmap1 = F.interpolate(featmap1, size=(self.h, self.w), mode='bilinear', align_corners=True)
            pts1_i = unproject_pts_pt(self.intrinsic1, coord.reshape(-1, 2), depth_layers1[i].flatten())
            pts1_i = pts1_i.reshape(1, self.h, self.w, 3)

            flow_inpainted, mask_no_mutual_flow_ = self.masked_diffuse(flow_f.permute(0, 3, 1, 2),
                                                                       mask_no_mutual_flow,
                                                                       kernel_size=15, iter=7)

            coord_inpainted = coord.clone()
            coord_inpainted_ = coord + flow_inpainted.permute(0, 2, 3, 1)
            mask_no_mutual_flow_bool = (mask_no_mutual_flow_ > 1e-6).squeeze(1)
            coord_inpainted[mask_no_mutual_flow_bool] = coord_inpainted_[mask_no_mutual_flow_bool]

            depth_inpainted = depth_layers1[i].clone()
            depth_inpainted_, mask_gau_i_ = self.masked_diffuse(depth2_sampled, mask_gau_i,
                                                                kernel_size=15, iter=7)
            mask_gau_i_bool = (mask_gau_i_ > 1e-6).squeeze(1)
            depth_inpainted.squeeze(1)[mask_gau_i_bool] = depth_inpainted_.squeeze(1)[mask_gau_i_bool]
            pts2_i = unproject_pts_pt(self.intrinsic2, coord_inpainted.contiguous().reshape(-1, 2),
                                    depth_inpainted.flatten()).reshape(1, self.h, self.w, 3)

            if i > 0:
                mask_wrong_ordering = (pts2_i[..., -1] <= last_pts2_i[..., -1]) * last_alpha_i
                pts2_i[mask_wrong_ordering] = last_pts2_i[mask_wrong_ordering] * 1.01

            rgba_end = mask_gau_i * torch.cat([rgb2_sampled, mask_gau_i], dim=1) + (1 - mask_gau_i) * rgba_i
            feat_end = mask_gau_i * featmap2_sampled + (1 - mask_gau_i) * featmap1
            last_alpha_i[alpha_i] = True
            last_pts2_i[alpha_i] = pts2_i[alpha_i]

            if with_inpainted:
                mask_keep = alpha_i
            else:
                mask_keep = mask_i.squeeze(1).bool()

            all_pts.append(pts1_i[mask_keep])
            all_rgbas.append(rgba_i.permute(0, 2, 3, 1)[mask_keep])
            all_feats.append(featmap1.permute(0, 2, 3, 1)[mask_keep])
            all_masks.append(mask_i.permute(0, 2, 3, 1)[mask_keep])
            all_pts_end.append(pts2_i[mask_keep])
            all_rgbas_end.append(rgba_end.permute(0, 2, 3, 1)[mask_keep])
            all_feats_end.append(feat_end.permute(0, 2, 3, 1)[mask_keep])
            all_optical_flows.append(flow_inpainted.permute(0, 2, 3, 1)[mask_keep])

        return all_pts, all_pts_end, all_rgbas, all_rgbas_end, all_feats, all_feats_end, all_masks, all_optical_flows

    def render_rgbda_layers_with_scene_flow(self, return_pts=False):
        kernel = torch.ones(5, 5, device=self.device)
        flow_f = self.scene_flow_estimator.compute_optical_flow(self.src_img1, self.src_img2)
        flow_b = self.scene_flow_estimator.compute_optical_flow(self.src_img2, self.src_img1)

        depth_bins1 = get_depth_bins(depth=self.src_depth1)
        depth_bins2 = get_depth_bins(depth=self.src_depth2)

        rgba_layers1, depth_layers1, mask_layers1 = \
            self.inpainter.sequential_inpainting(self.src_img1, self.src_depth1, depth_bins1)
        rgba_layers2, depth_layers2, mask_layers2 = \
            self.inpainter.sequential_inpainting(self.src_img2, self.src_depth2, depth_bins2)
        if self.args.visualize_rgbda_layers:
            self.save_rgbda_layers(self.src_rgb_file1, rgba_layers1, depth_layers1, mask_layers1)
            self.save_rgbda_layers(self.src_rgb_file2, rgba_layers2, depth_layers2, mask_layers2)

        featmaps1 = self.feature_extraction(rgba_layers1, mask_layers1, depth_layers1)
        featmaps2 = self.feature_extraction(rgba_layers2, mask_layers2, depth_layers2)

        depth_layers1 = self.apply_scale_shift(depth_layers1, self.scale_shift1[0], self.scale_shift1[1])
        depth_layers2 = self.apply_scale_shift(depth_layers2, self.scale_shift2[0], self.scale_shift2[1])

        processed_depth1, layer_id_map1 = self.render_depth_from_mdi(depth_layers1, rgba_layers1[:, :, -1:])
        processed_depth2, layer_id_map2 = self.render_depth_from_mdi(depth_layers2, rgba_layers2[:, :, -1:])

        assert self.src_img1.shape[-2:] == self.src_img2.shape[-2:]
        h, w = self.src_img1.shape[-2:]
        coord = get_coord_grids_pt(h, w, device=self.device).float()[None]
        pts1 = unproject_pts_pt(self.intrinsic1, coord.reshape(-1, 2), processed_depth1.flatten())
        pts2 = unproject_pts_pt(self.intrinsic2, coord.reshape(-1, 2), processed_depth2.flatten())

        all_pts_11, all_pts_12, all_rgbas_11, all_rgbas_12, all_feats_11, all_feats_12,\
        all_masks_1, all_optical_flow_1 = \
            self.compute_scene_flow_one_side(coord, torch.inverse(self.pose), self.src_img1, self.src_img2,
                                             rgba_layers1, rgba_layers2, featmaps1, featmaps2,
                                             pts1, pts2, depth_layers1, depth_layers2, mask_layers1, mask_layers2,
                                            flow_f, flow_b, kernel, with_inpainted=True)

        all_pts_22, all_pts_21, all_rgbas_22, all_rgbas_21, all_feats_22, all_feats_21,\
        all_masks_2, all_optical_flow_2 = \
            self.compute_scene_flow_one_side(coord, self.pose, self.src_img2, self.src_img1,
                                             rgba_layers2, rgba_layers1, featmaps2, featmaps1,
                                             pts2, pts1, depth_layers2, depth_layers1, mask_layers2, mask_layers1,
                                             flow_b, flow_f, kernel, with_inpainted=True)

        if not torch.allclose(self.pose, torch.eye(4, device=self.device)):
            all_pts_21 = self.transform_all_pts(all_pts_21, torch.inverse(self.pose))
            all_pts_22 = self.transform_all_pts(all_pts_22, torch.inverse(self.pose))

        all_pts = torch.cat(all_pts_11+all_pts_21)
        all_rgbas = torch.cat(all_rgbas_11+all_rgbas_21)
        all_feats = torch.cat(all_feats_11+all_feats_21)
        all_masks = torch.cat(all_masks_1+all_masks_2)
        all_pts_end = torch.cat(all_pts_12+all_pts_22)
        all_rgbas_end = torch.cat(all_rgbas_12+all_rgbas_22)
        all_feats_end = torch.cat(all_feats_12+all_feats_22)
        all_side_ids = torch.zeros_like(all_masks.squeeze(), dtype=torch.long)
        num_pts_2 = sum([len(x) for x in all_pts_21])
        all_side_ids[-num_pts_2:] = 1
        all_optical_flow = torch.cat(all_optical_flow_1+all_optical_flow_2)

        if return_pts:
            return all_pts, all_pts_end, all_rgbas, all_rgbas_end, all_feats, all_feats_end, \
                   all_masks, all_side_ids, all_optical_flow

        R = self.tgt_pose[0, :3, :3]
        t = self.tgt_pose[0, :3, 3]
        pred_img, direct_rgb_out, meta = self.render_pcd(all_pts, all_pts_end,
                                                         all_rgbas, all_rgbas_end,
                                                         all_feats, all_feats_end,
                                                         all_masks, all_side_ids,
                                                         R, t, self.time)
        mask1 = self.get_reprojection_mask(pts1, R, t)
        pose2_to_tgt = self.tgt_pose.bmm(torch.inverse(self.pose))
        mask2 = self.get_reprojection_mask(pts2, pose2_to_tgt[0, :3, :3], pose2_to_tgt[0, :3, 3])
        mask = (mask1+mask2) * 0.5
        gt_img = self.tgt_img
        t, b, l, r, bad = self.get_cropping_ids(mask)
        skip = False
        if not skip and not self.args.eval_mode:
            pred_img = pred_img[:, :, t:b, l:r]
            mask = mask[:, :, t:b, l:r]
            direct_rgb_out = direct_rgb_out[:, :, t:b, l:r]
            gt_img = gt_img[:, :, t:b, l:r]
        else:
            skip = True

        res_dict = {
            'src_img1': self.src_img1,
            'src_img2': self.src_img2,
            'pred_img': pred_img,
            'gt_img': gt_img,
            'mask': mask,
            'direct_rgb_out': direct_rgb_out,
            'alpha_layers1': rgba_layers1[:, :, [-1]],
            'alpha_layers2': rgba_layers2[:, :, [-1]],
            'mask_layers1': mask_layers1,
            'mask_layers2': mask_layers2,
            'skip': skip
        }
        return res_dict

    def dynamic_view_synthesis_with_inpainting(self):
        if self.is_multi_view:
            return self.render_rgbda_layers_from_one_view()
        else:
            return self.render_rgbda_layers_with_scene_flow()

    def get_prediction(self, data):
        # process data first
        self.process_data(data)
        return self.dynamic_view_synthesis_with_inpainting()

    def save_rgbda_layers(self, src_rgb_file, rgba_layers, depth_layers, mask_layers):
        frame_id = os.path.basename(src_rgb_file).split('.')[0]
        scene_id = src_rgb_file.split('/')[-3]
        out_dir = os.path.join(self.args.rootdir, 'out', self.args.expname, 'vis',
                               '{}-{}'.format(scene_id, frame_id))
        os.makedirs(out_dir, exist_ok=True)

        alpha_layers = rgba_layers[:, :, [-1]]
        for i, rgba_layer in enumerate(rgba_layers):
            save_filename = os.path.join(out_dir, 'rgb_original_{}.png'.format(i))
            rgba_layer_ = rgba_layer * mask_layers[i]
            rgba_np = rgba_layer_.detach().squeeze().permute(1, 2, 0).cpu().numpy()
            imageio.imwrite(save_filename, float2uint8(rgba_np))

        for i, rgba_layer in enumerate(rgba_layers):
            save_filename = os.path.join(out_dir, 'rgb_{}.png'.format(i))
            rgba_np = rgba_layer.detach().squeeze().permute(1, 2, 0).cpu().numpy()
            imageio.imwrite(save_filename, float2uint8(rgba_np))

        for i, depth_layer in enumerate(depth_layers):
            save_filename = os.path.join(out_dir, 'disparity_original_{}.png'.format(i))
            disparity = (1. / torch.clamp(depth_layer, min=1e-6)) * alpha_layers[i]
            disparity = torch.cat([disparity, disparity, disparity, alpha_layers[i]*mask_layers[i]], dim=1)
            disparity_np = disparity.detach().squeeze().cpu().numpy().transpose(1, 2, 0)
            imageio.imwrite(save_filename, float2uint8(disparity_np))

        for i, depth_layer in enumerate(depth_layers):
            save_filename = os.path.join(out_dir, 'disparity_{}.png'.format(i))
            disparity = (1. / torch.clamp(depth_layer, min=1e-6)) * alpha_layers[i]
            disparity = torch.cat([disparity, disparity, disparity, alpha_layers[i]], dim=1)
            disparity_np = disparity.detach().squeeze().cpu().numpy().transpose(1, 2, 0)
            imageio.imwrite(save_filename, float2uint8(disparity_np))

        for i, mask_layer in enumerate(mask_layers):
            save_filename = os.path.join(out_dir, 'mask_{}.png'.format(i))
            tri_mask = 0.5 * alpha_layers[i] + 0.5 * mask_layer
            tri_mask_np = tri_mask.detach().squeeze().cpu().numpy()
            imageio.imwrite(save_filename, float2uint8(tri_mask_np))



