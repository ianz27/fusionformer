# train
## fusion of camera and lidar
now we fuse camera only([detr3d-r50](https://drive.google.com/drive/folders/18q2sQ-J-AxqeCO8FaAWKQ9Fi13PPv_MR)) and lidar only([centerpoint](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/centerpoint/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20201001_135205-5db91e00.pth)) pretrained model for fusion pretrained
```shell
python tools/model_converters/fuse_model.py
```
```shell
tools/dist_train.sh configs/fusionformer/base_fusion_test.py gpu_num
```
## camera only
```shell
tools/dist_train.sh configs/detr3d/detr3d_r50.py gpu_num
```
## lidar only
```shell
tools/dist_train.sh configs/fusionformer/base_lidar_only.py gpu_num
```