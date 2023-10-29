import torch

img_ckpt = torch.load('work_dirs/detr3d_r50/epoch_24.pth')
state_dict1 = img_ckpt['state_dict']

pts_ckpt = torch.load('pth/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20201001_135205-5db91e00.pth')
state_dict2 = pts_ckpt['state_dict']
# pts_head in camera checkpoint will be overwrite by lidar checkpoint
state_dict1.update(state_dict2)

merged_state_dict = state_dict1

save_checkpoint = {'state_dict':merged_state_dict }

torch.save(save_checkpoint, 'pth/c_detr3d_r50_l_centerpoint.pth')
