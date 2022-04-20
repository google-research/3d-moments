#!/bin/bash
# download DPT (https://github.com/isl-org/DPT) pretrained weights into DPT/weights
wget https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt -P DPT/weights

# download RAFT (https://github.com/princeton-vl/RAFT) pretrained weights into RAFT/models/
wget https://www.dropbox.com/s/4j4z58wuv8o0mfz/models.zip
unzip models.zip -d RAFT/
rm -rf models.zip

# download rgbd inpainting pretrained weights into inpainting_ckpts/
wget https://filebox.ece.vt.edu/~jbhuang/project/3DPhoto/model/color-model.pth
wget https://filebox.ece.vt.edu/~jbhuang/project/3DPhoto/model/depth-model.pth
wget https://filebox.ece.vt.edu/~jbhuang/project/3DPhoto/model/edge-model.pth

mv color-model.pth inpainting_ckpts/
mv depth-model.pth inpainting_ckpts/
mv edge-model.pth inpainting_ckpts/

# download the 3D moments pretrained model:
gdown https://drive.google.com/uc?id=1alRujVVWhqysU2xD8i5JvEvkVYinxLLD -O pretrained/