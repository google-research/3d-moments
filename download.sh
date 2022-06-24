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


# download DPT (https://github.com/isl-org/DPT) pretrained weights into DPT/weights
wget https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt -P third_party/DPT/weights

# download RAFT (https://github.com/princeton-vl/RAFT) pretrained weights into RAFT/models/
wget https://www.dropbox.com/s/4j4z58wuv8o0mfz/models.zip
unzip models.zip -d third_party/RAFT/
rm -rf models.zip

# download rgbd inpainting pretrained weights into inpainting_ckpts/
wget https://filebox.ece.vt.edu/~jbhuang/project/3DPhoto/model/color-model.pth
wget https://filebox.ece.vt.edu/~jbhuang/project/3DPhoto/model/depth-model.pth
wget https://filebox.ece.vt.edu/~jbhuang/project/3DPhoto/model/edge-model.pth
mkdir inpainting_ckpts/
mv color-model.pth inpainting_ckpts/
mv depth-model.pth inpainting_ckpts/
mv edge-model.pth inpainting_ckpts/

# download the 3D moments pretrained model:
gdown https://drive.google.com/uc?id=1keqdnl2roBO2MjXhbd0VbfYaGAhyjUed -O pretrained/
