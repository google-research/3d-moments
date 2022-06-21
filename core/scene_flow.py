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


from third_party.RAFT.core.utils.utils import InputPadder
from core.utils import *


class SceneFlowEstimator():
    def __init__(self, args, model):
        device = "cuda:{}".format(args.local_rank)
        self.device = device
        self.raft_model = model
        self.train_raft = args.train_raft

    def compute_optical_flow(self, img1, img2, return_np_array=False):
        '''
        :param img1: [B, 3, H, W]
        :param img2: [B, 3, H, W]
        :return: optical_flow, [B, H, W, 2]
        '''
        if not self.train_raft:
            with torch.no_grad():
                assert img1.max() <= 1 and img2.max() <= 1
                image1 = img1 * 255.
                image2 = img2 * 255.
                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)

                flow_low, flow_up = self.raft_model.module(image1, image2, iters=20, test_mode=True, padder=padder)

                if return_np_array:
                    return flow_up.cpu().numpy().transpose(0, 2, 3, 1)

                return flow_up.permute(0, 2, 3, 1).detach()  # [B, h, w, 2]
        else:
            assert img1.max() <= 1 and img2.max() <= 1
            image1 = img1 * 255.
            image2 = img2 * 255.
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            flow_predictions = self.raft_model.module(image1, image2, iters=20, padder=padder)
            return flow_predictions[-1].permute(0, 2, 3, 1)  # [B, h, w, 2]

    def get_mutual_matches(self, flow_f, flow_b, th=2., return_mask=False):
        assert flow_f.shape == flow_b.shape
        batch_size = flow_f.shape[0]
        assert flow_f.shape[1:3] == flow_b.shape[1:3]
        h, w = flow_f.shape[1:3]
        grid = get_coord_grids_pt(h, w, self.device)[None].float().repeat(batch_size, 1, 1, 1)  # [B, h, w, 2]
        grid2 = grid + flow_f
        mask_boundary = (grid2[..., 0] >= 0) * (grid2[..., 0] <= w - 1) * \
                        (grid2[..., 1] >= 0) * (grid2[..., 1] <= h - 1)
        grid2_normed = normalize_for_grid_sample(grid2, h, w)
        flow_b_sampled = F.grid_sample(flow_b.permute(0, 3, 1, 2), grid2_normed,
                                       align_corners=True).permute(0, 2, 3, 1)
        grid1 = grid2 + flow_b_sampled
        mask_boundary *= (grid1[..., 0] >= 0) * (grid1[..., 0] <= w - 1) * \
                         (grid1[..., 1] >= 0) * (grid1[..., 1] <= h - 1)

        fb_map = flow_f + flow_b_sampled
        mask_valid = mask_boundary * (torch.norm(fb_map, dim=-1) < th)
        if return_mask:
            return mask_valid
        coords1 = grid[mask_valid]  # [n, 2]
        coords2 = grid2[mask_valid]  # [n, 2]
        return coords1, coords2