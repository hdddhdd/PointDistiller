import torch
from mmcv import Config
from mmdet3d.models import build_detector
from mmdet3d.models.detectors.voxelnet import VoxelNet

def forward_dummy(self, points):
    voxels, num_points, coords = self.voxelize(points)

    print(f"voxels shape: {voxels.shape}")
    print(f"num_points shape: {num_points.shape}")
    print(f"coords shape: {coords.shape}")

    voxel_features = self.voxel_encoder(voxels, num_points, coords)

    x = self.middle_encoder(voxel_features, coords)

    # backbone에 list가 아닌 tensor만 전달하도록 보정
    if isinstance(x, list):
        x = x[0]

    x = self.backbone(x)

    if self.with_neck:
        x = self.neck(x)

    outs = self.bbox_head(x)
    return outs

VoxelNet.forward_dummy = forward_dummy


def count_flops_and_params(model, input_tensor):
    from fvcore.nn import FlopCountAnalysis, parameter_count
    model.eval()

    with torch.no_grad():
        flops = FlopCountAnalysis(model, input_tensor)
        params = parameter_count(model)
    
    return flops.total(), params[""]

def main():
    # cfg_path = '../../configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_64x.py'
    # cfg_path = '../../configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py'
    cfg_path = './configs/pointpillars/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d.py'
    # cfg_path = './configs/pointpillars/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_student_2x.py'

    cfg = Config.fromfile(cfg_path)
    cfg.model.train_cfg = None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    model.to(device)
    model.eval()

    # Sparse ops used by CenterPoint (voxelization / sparse conv) are CUDA-only,
    # so FLOPs calculation must run on GPU.
    # Use the model's expected point feature size.
    num_point_channels = cfg.model.pts_voxel_encoder.get('in_channels', 4)
    dummy_points = [
        torch.rand(40000, num_point_channels, device=device).float()
    ]

    # ✅ 핵심 수정: forward를 forward_dummy로 덮어쓰기
    model.forward_dummy = model.forward_dummy.__get__(model)
    model.forward = model.forward_dummy

    from fvcore.nn import FlopCountAnalysis, parameter_count
    flop_analyzer = FlopCountAnalysis(model, dummy_points)
    flops = flop_analyzer.total()
    params = parameter_count(model)

    print(f'FLOPs: {flops / 1e9:.2f} GFLOPs')
    print(f'Params: {params[""] / 1e6:.2f} M')



if __name__ == '__main__':
    main()
