import torch
import torch.cuda
from torch.autograd import Variable
from skimage.color import (rgb2lab, rgb2yuv, rgb2ycbcr, lab2rgb, yuv2rgb, ycbcr2rgb,
                           rgb2hsv, hsv2rgb, rgb2xyz, xyz2rgb, rgb2hed, hed2rgb)


def _convert(input_, type_):
    return {
        'float': input_.float(),
        'double': input_.double(),
    }.get(type_, input_)


def _generic_transform_sk_4d(transform, in_type='', out_type=''):
    def apply_transform(input_):
        to_squeeze = (input_.dim() == 3)
        device = input_.device
        input_ = input_.cpu()
        input_ = _convert(input_, in_type)

        if to_squeeze:
            input_ = input_.unsqueeze(0)

        input_ = input_.permute(0, 2, 3, 1).numpy()
        transformed = transform(input_)
        output = torch.from_numpy(transformed).float().permute(0, 3, 1, 2)
        if to_squeeze:
            output = output.squeeze(0)
        output = _convert(output, out_type)
        return output.to(device)
    return apply_transform


def _generic_transform_sk_3d(transform, in_type='', out_type=''):
    def apply_transform_individual(input_):
        device = input_.device
        input_ = input_.cpu()
        input_ = _convert(input_, in_type)

        input_ = input_.permute(1, 2, 0).numpy()
        transformed = transform(input_)
        output = torch.from_numpy(transformed).float().permute(2, 0, 1)
        output = _convert(output, out_type)
        return output.to(device)

    def apply_transform(input_):
        to_stack = []
        for image in input_:
            to_stack.append(apply_transform_individual(image))
        return torch.stack(to_stack)
    return apply_transform


# --- Cie*LAB ---
rgb_to_lab = _generic_transform_sk_4d(rgb2lab)
lab_to_rgb = _generic_transform_sk_3d(lab2rgb, in_type='double', out_type='float')
# --- YUV ---
rgb_to_yuv = _generic_transform_sk_4d(rgb2yuv)
yuv_to_rgb = _generic_transform_sk_4d(yuv2rgb)
# --- YCbCr ---
rgb_to_ycbcr = _generic_transform_sk_4d(rgb2ycbcr)
ycbcr_to_rgb = _generic_transform_sk_4d(ycbcr2rgb, in_type='double', out_type='float')
# --- HSV ---
rgb_to_hsv = _generic_transform_sk_3d(rgb2hsv)
hsv_to_rgb = _generic_transform_sk_3d(hsv2rgb)
# --- XYZ ---
rgb_to_xyz = _generic_transform_sk_4d(rgb2xyz)
xyz_to_rgb = _generic_transform_sk_3d(xyz2rgb, in_type='double', out_type='float')
# --- HED ---
rgb_to_hed = _generic_transform_sk_4d(rgb2hed)
hed_to_rgb = _generic_transform_sk_3d(hed2rgb, in_type='double', out_type='float')


def err(type_):
    raise NotImplementedError('Color space conversion %s not implemented yet' % type_)


def convert(input_, type_):
    return {
        'rgb2lab': rgb_to_lab(input_),
        'lab2rgb': lab_to_rgb(input_),
        'rgb2yuv': rgb_to_yuv(input_),
        'yuv2rgb': yuv_to_rgb(input_),
        'rgb2xyz': rgb_to_xyz(input_),
        'xyz2rgb': xyz_to_rgb(input_),
        'rgb2hsv': rgb_to_hsv(input_),
        'hsv2rgb': hsv_to_rgb(input_),
        'rgb2ycbcr': rgb_to_ycbcr(input_),
        'ycbcr2rgb': ycbcr_to_rgb(input_)
    }.get(type_, err(type_))
