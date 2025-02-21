"""
This file contains helper functions used in this project
"""
import numpy as np
import torch
import os
import shutil
import torch
from tifffile import imwrite
from collections import defaultdict
import pathlib

from torchvision import transforms


def minmax_normalization(input_):
    """
    This functon normalize the input to range of 0-1

    input_: the input
    """
    if isinstance(input_, np.ndarray):
        input_ = (input_ - np.amin(input_)) / (np.amax(input_) - np.amin(input_))

    elif isinstance(input_, torch.Tensor):
        # pylint: disable=no-member
        input_ = (input_ - torch.min(input_)) / (torch.max(input_) - torch.min(input_))
        # pylint: enable=no-member

    else:
        raise NotImplementedError("expected numpy or torch array")

    return input_

# from ray import tune


def torch_complex_normalize(x):
    x_angle = torch.angle(x)
    x_abs = torch.abs(x)

    x_abs -= torch.min(x_abs)
    x_abs /= torch.max(x_abs)

    x = x_abs * np.exp(1j * x_angle)

    return x


def strip_empties_from_dict(data):
    new_data = {}
    for k, v in data.items():
        if isinstance(v, dict):
            v = strip_empties_from_dict(v)

        if v not in (None, str(), list(), dict(),):
            new_data[k] = v
    return new_data


def ray_tune_override_config_from_param_space(config, param_space):
    for k in param_space:
        if isinstance(param_space[k], dict):
            ray_tune_override_config_from_param_space(config[k], param_space[k])

        else:
            config[k] = param_space[k]

    return config


def get_last_folder(path):
    return pathlib.PurePath(path).name


def convert_pl_outputs(outputs):
    outputs_dict = defaultdict(list)

    for i in range(len(outputs)):
        for k in outputs[i]:
            outputs_dict[k].append(outputs[i][k])

    log_dict, img_dict = {}, {}
    for k in outputs_dict:
        try:
            tmp = torch.Tensor(outputs_dict[k]).detach().cpu()

            log_dict.update({
                k: tmp
            })

        except Exception:
            if outputs_dict[k][0].dim() == 2:
                tmp = torch.stack(outputs_dict[k], 0).detach().cpu()
            else:
                tmp = torch.cat(outputs_dict[k], 0).detach().cpu()

            if tmp.dtype == torch.complex64:
                tmp = torch.abs(tmp)

            img_dict.update({
                k: tmp
            })

    return log_dict, img_dict


def check_and_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def copy_code_to_path(src_path=None, file_path=None):
    if (file_path is not None) and (src_path is not None):
        check_and_mkdir(file_path)

        max_code_save = 100  # only 100 copies can be saved
        for i in range(max_code_save):
            code_path = os.path.join(file_path, 'code%d/' % i)
            # print(f"src_path: {src_path}\ncode_path: {code_path}")
            if not os.path.exists(code_path):
                shutil.copytree(src=src_path, dst=code_path)
                break


def merge_child_dict(d, ret, prefix=''):

    for k in d:
        if k in ['setting', 'test']:
            continue

        if isinstance(d[k], dict):
            merge_child_dict(d[k], ret=ret, prefix= prefix + k + '/')
        else:
            ret.update({
                prefix + k: d[k]
            })

    return ret


def write_test(save_path, log_dict=None, img_dict=None):

    if log_dict:

        cvs_data = torch.stack([log_dict[k] for k in log_dict], 0).numpy()
        cvs_data = np.transpose(cvs_data, [1, 0])

        cvs_data_mean = cvs_data.mean(0)
        cvs_data_mean.shape = [1, -1]

        cvs_data_std = cvs_data.std(0)
        cvs_data_std.shape = [1, -1]

        cvs_data_min = cvs_data.min(0)
        cvs_data_min.shape = [1, -1]

        cvs_data_max = cvs_data.max(0)
        cvs_data_max.shape = [1, -1]

        num_index = cvs_data.shape[0]
        cvs_index = np.arange(num_index) + 1
        cvs_index.shape = [-1, 1]

        cvs_data_with_index = np.concatenate([cvs_index, cvs_data], 1)

        cvs_header = ''
        for k in log_dict:
            cvs_header = cvs_header + k + ','

        np.savetxt(os.path.join(save_path, 'metrics.csv'), cvs_data_with_index,
                   delimiter=',', fmt='%.5f', header='index,' + cvs_header)
        np.savetxt(os.path.join(save_path, 'metrics_mean.csv'), cvs_data_mean,
                   delimiter=',', fmt='%.5f', header=cvs_header)
        np.savetxt(os.path.join(save_path, 'metrics_std.csv'), cvs_data_std,
                   delimiter=',', fmt='%.5f',  header=cvs_header)
        np.savetxt(os.path.join(save_path, 'metrics_min.csv'), cvs_data_min,
                   delimiter=',', fmt='%.5f', header=cvs_header)
        np.savetxt(os.path.join(save_path, 'metrics_max.csv'), cvs_data_max,
                   delimiter=',', fmt='%.5f', header=cvs_header)

        print("==========================")
        print("HEADER:", cvs_header)
        print("MEAN:", cvs_data_mean)
        print("STD:", cvs_data_std)
        print("MAX:", cvs_data_max)
        print("MIN:", cvs_data_min)
        print("==========================")

    if img_dict:

        for k in img_dict:

            imwrite(os.path.join(save_path, k + '.tiff'), data=np.array(img_dict[k]), imagej=True)


def get_save_path_from_config(config):
    # save_path = os.path.join(config['setting']['exp_path'], config['setting']['exp_folder'])
    save_path = os.path.join(config['setting']['save_dir'])
    check_and_mkdir(save_path)

    return save_path


def abs_helper(x, axis=1, is_normalization=True):
    x = torch.sqrt(torch.sum(x ** 2, dim=axis, keepdim=True))

    if is_normalization:
        for i in range(x.shape[0]):
            x[i] = (x[i] - torch.min(x[i])) / (torch.max(x[i]) - torch.min(x[i]) + 1e-16)

    x = x.to(torch.float32)

    return x


def crop_images(images, crop_width):
    # raise ValueError(f"images.shape: {images.shape}")
    if len(images.shape) != 4:
        raise ValueError("Check the size of the images")
    # crop_width = min(images.shape[2], images.shape[3])

    cropped_images = []

    for i in range(images.shape[0]):  # Iterate over the number of images
        img = images[i]  # Select one image
        for j in range(images.shape[1]):  # Iterate over the channels
            channel = img[j]  # Select one channel
            cropping_transform = transforms.CenterCrop((crop_width, crop_width))
            cropped_channel = cropping_transform(channel)
            cropped_images.append(cropped_channel)

    # Reshape the cropped images
    cropped_images = torch.stack(cropped_images, dim=0)
    cropped_images = cropped_images.view(images.shape[0], images.shape[1], crop_width, crop_width)

    return cropped_images


def crop_images_del(images, crop_width, resize_and_crop=True):
    if len(images.shape) != 4:
        raise ValueError("Check the size of the images")
    batch_size, channels, height, width = images.shape
    resized_images = []
    crop_transform = transforms.CenterCrop((crop_width, crop_width))

    for i in range(batch_size):  # Iterate over the batch
        img = images[i]  # Single image with shape (C, H, W)

        # Determine scale factor to resize
        scale_factor = crop_width / min(height, width)
        new_height, new_width = int(height * scale_factor), int(width * scale_factor)

        # Resize each channel independently
        resized_channels = []
        resize_transform = transforms.Resize((new_height, new_width))
        for c in range(channels):
            resized_channel = resize_transform(img[c].unsqueeze(0))  # Add batch dim
            resized_channels.append(resized_channel.squeeze(0))  # Remove batch dim
        resized_img = torch.stack(resized_channels, dim=0)  # Shape: (C, new_height, new_width)

        # Crop to desired size
        cropped_img = crop_transform(resized_img)
        resized_images.append(cropped_img)
    
    # Stack the resized and cropped images back into a batch
    cropped_images = torch.stack(resized_images, dim=0)

    print(f"Output images shape: {cropped_images.shape}")
    return cropped_images