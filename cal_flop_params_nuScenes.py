# python cal_flop_params.py configs/centerpoint/centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_nus.py --device cuda --show-unsupported --show-uncalled
import argparse
from collections import Counter
from typing import Iterable

import torch
from mmcv import Config, DictAction

from mmdet3d.models import build_model

try:
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn.jit_handles import get_shape
except ImportError as err:
    raise ImportError('Please install fvcore: pip install fvcore') from err


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute FLOPs/params for MMDetection3D models')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--num-points',
        type=int,
        default=40000,
        help='number of points per sample for dummy input')
    parser.add_argument(
        '--point-dim',
        type=int,
        default=None,
        help='point feature dimension override (auto-infer if omitted)')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='batch size for dummy input')
    parser.add_argument(
        '--device',
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='device for profiling (default: cuda if available)')
    parser.add_argument(
        '--show-unsupported',
        action='store_true',
        help='print unsupported operators from fvcore tracer')
    parser.add_argument(
        '--show-uncalled',
        action='store_true',
        help='print modules not used by forward_dummy')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override settings in the config, key=value')
    return parser.parse_args()


def _numel(value) -> int:
    shape = get_shape(value)
    if shape is None:
        return 0
    if len(shape) == 0:
        return 1
    total = 1
    for dim in shape:
        if not isinstance(dim, int) or dim < 0:
            return 0
        total *= dim
    return int(total)


def _zero_flop_jit(inputs, outputs):
    return Counter({'flops': 0})


def _elementwise_flop_jit(inputs, outputs):
    flops = sum(_numel(out) for out in outputs)
    return Counter({'flops': flops})


def _sum_flop_jit(inputs, outputs):
    if not inputs:
        return Counter({'flops': 0})
    input_elems = _numel(inputs[0])
    output_elems = sum(_numel(out) for out in outputs)
    # A reduction sum roughly costs N - 1 adds per reduced tensor.
    flops = max(input_elems - output_elems, 0)
    return Counter({'flops': flops})


def _infer_point_dim(cfg) -> int:
    model_cfg = cfg.model
    voxel_encoder_cfg = (
        model_cfg.get('pts_voxel_encoder', None)
        or model_cfg.get('voxel_encoder', None))
    if voxel_encoder_cfg is not None:
        if 'num_features' in voxel_encoder_cfg:
            return int(voxel_encoder_cfg['num_features'])
        if 'in_channels' in voxel_encoder_cfg:
            return int(voxel_encoder_cfg['in_channels'])

    data_cfg = cfg.get('data', {})
    test_cfg = data_cfg.get('test', {})
    pipeline = test_cfg.get('pipeline', [])
    if pipeline:
        for step in pipeline:
            if step.get('type') == 'LoadPointsFromFile':
                use_dim = step.get('use_dim', None)
                if isinstance(use_dim, int):
                    return int(use_dim)
                if isinstance(use_dim, (list, tuple)):
                    return len(use_dim)
                load_dim = step.get('load_dim', None)
                if isinstance(load_dim, int):
                    return int(load_dim)

    return 4


def _ensure_forward_dummy(model):
    # Some branches in this repo define broken forward_dummy implementations
    # (e.g., VoxelNet uses pts_* names that do not exist). Bind a safe one.
    has_mvx_path = all(
        hasattr(model, attr) for attr in (
            'voxelize',
            'pts_voxel_encoder',
            'pts_middle_encoder',
            'pts_backbone',
            'pts_bbox_head'))
    has_voxelnet_path = all(
        hasattr(model, attr) for attr in (
            'voxelize',
            'voxel_encoder',
            'middle_encoder',
            'backbone',
            'bbox_head'))

    if has_mvx_path:

        def _safe_forward_dummy(self, points):
            voxels, num_points, coors = self.voxelize(points)
            voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
            batch_size = int(coors[-1, 0]) + 1 if coors.numel() > 0 else len(points)
            x = self.pts_middle_encoder(voxel_features, coors, batch_size)
            x = self.pts_backbone(x)
            if getattr(self, 'with_pts_neck', False):
                x = self.pts_neck(x)
            return self.pts_bbox_head(x)

    elif has_voxelnet_path:

        def _safe_forward_dummy(self, points):
            voxels, num_points, coors = self.voxelize(points)
            voxel_features = self.voxel_encoder(voxels, num_points, coors)
            batch_size = int(coors[-1, 0]) + 1 if coors.numel() > 0 else len(points)
            x = self.middle_encoder(voxel_features, coors, batch_size)
            x = self.backbone(x)
            if getattr(self, 'with_neck', False):
                x = self.neck(x)
            return self.bbox_head(x)

    elif hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
        return
    else:
        raise NotImplementedError(
            f'No safe forward_dummy path for {model.__class__.__name__}.')

    model.forward_dummy = _safe_forward_dummy.__get__(model)
    model.forward = model.forward_dummy


def _count_called_params(model, uncalled_modules: Iterable[str]) -> int:
    uncalled_modules = set(uncalled_modules)
    if not uncalled_modules:
        return sum(p.numel() for p in model.parameters())

    def from_uncalled_module(param_name: str) -> bool:
        module_name = param_name.rsplit('.', 1)[0] if '.' in param_name else ''
        for uncalled in uncalled_modules:
            if module_name == uncalled or module_name.startswith(f'{uncalled}.'):
                return True
        return False

    return sum(
        p.numel() for name, p in model.named_parameters()
        if not from_uncalled_module(name))


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if cfg.model.get('pretrained', None) is not None:
        cfg.model.pretrained = None

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()
    _ensure_forward_dummy(model)

    point_dim = args.point_dim if args.point_dim is not None else _infer_point_dim(cfg)
    dummy_points = [
        torch.rand(args.num_points, point_dim, device=device, dtype=torch.float32)
        for _ in range(args.batch_size)
    ]

    flop_analyzer = FlopCountAnalysis(model, (dummy_points,))
    flop_analyzer.uncalled_modules_warnings(False)
    flop_analyzer.unsupported_ops_warnings(False)

    for op_name in ('aten::add', 'aten::sub', 'aten::mul', 'aten::mul_', 'aten::div'):
        flop_analyzer.set_op_handle(op_name, _elementwise_flop_jit)
    flop_analyzer.set_op_handle('aten::sum', _sum_flop_jit)
    flop_analyzer.set_op_handle('aten::repeat', _zero_flop_jit)
    flop_analyzer.set_op_handle('prim::PythonOp._Voxelization', _zero_flop_jit)

    with torch.no_grad():
        total_flops = flop_analyzer.total()

    unsupported_ops = flop_analyzer.unsupported_ops()
    uncalled_modules = flop_analyzer.uncalled_modules()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    called_params = _count_called_params(model, uncalled_modules)

    split = '=' * 60
    print(split)
    print(f'Config: {args.config}')
    print(f'Device: {device}')
    print(
        f'Dummy input: batch={args.batch_size}, '
        f'points={args.num_points}, point_dim={point_dim}')
    print(f'FLOPs (forward_dummy): {total_flops / 1e9:.2f} GFLOPs')
    print(f'Params (all): {total_params / 1e6:.2f} M')
    print(f'Params (trainable): {trainable_params / 1e6:.2f} M')
    print(f'Params (called modules only): {called_params / 1e6:.2f} M')
    print(split)

    if args.show_unsupported:
        if unsupported_ops:
            print('Unsupported operators:')
            for op_name, count in unsupported_ops.items():
                print(f'  - {op_name}: {count}')
        else:
            print('Unsupported operators: none')

    if args.show_uncalled:
        if uncalled_modules:
            print('Uncalled modules:')
            for module_name in sorted(uncalled_modules):
                print(f'  - {module_name}')
        else:
            print('Uncalled modules: none')


if __name__ == '__main__':
    main()
