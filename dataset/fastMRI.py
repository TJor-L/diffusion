import os
import numpy as np
from glob import glob
import nibabel as nib
import torch
from nibabel.processing import conform
import ants
import h5py
import tqdm
import tifffile
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


import h5py
import os
import numpy as np
import tqdm
import torch
import tifffile
import pandas
from torch.utils.data import Dataset
import json

import matplotlib.pyplot as plt
from utility.utility import crop_images, abs_helper
from utility.tweedie_utility import compute_metrics

ROOT_PATH = '/project/cigserver4/export2/Dataset/fastmri_brain_multicoil'
DATA_PATH = '/project/cigserver4/export2/Dataset/fastmri_brain_multicoil'
#ROOT_PATH = '/export/project/p.youngil/fastmri'
#DATASHEET = pandas.read_csv(os.path.join(ROOT_PATH, 'fastmri_brain_multicoil_20230102.csv'))
DATASHEET = pandas.read_csv(os.path.join('/project/cigserver4/export2/Dataset/fastmri_brain_multicoil', 'fastmri_brain_multicoil_20230102.csv'))

ALL_IDX_LIST = list(DATASHEET['INDEX'])


def INDEX2_helper(idx, key_):
    file_id_df = DATASHEET[key_][DATASHEET['INDEX'] == idx]

    assert len(file_id_df.index) == 1

    return file_id_df[idx]


INDEX2FILE = lambda idx: INDEX2_helper(idx, 'FILE')


def INDEX2DROP(idx):
    ret = INDEX2_helper(idx, 'DROP')

    if ret in ['0', 'false', 'False', 0.0]:
        return False
    else:
        return True


def INDEX2SLICE_START(idx):
    ret = INDEX2_helper(idx, 'SLICE_START')

    if isinstance(ret, np.float64) and ret >= 0:
        return int(ret)
    else:
        return None


def INDEX2SLICE_END(idx):
    ret = INDEX2_helper(idx, 'SLICE_END')

    if isinstance(ret, np.float64) and ret >= 0:
        return int(ret)
    else:
        return None


def ftran(y, smps, mask):
    """
    compute adjoint of fast MRI, x = smps^H F^H mask^H x

    :param y: under-sampled measurements, shape: batch, coils, width, height; dtype: complex
    :param smps: sensitivity maps, shape: batch, coils, width, height; dtype: complex
    :param mask: sampling mask, shape: batch, width, height; dtype: float/bool
    :return: zero-filled image
    """
    
    """
    y.shape: 1, 20, 320, 320
    mask.shape: 1, 320, 320
    """

    y = y * mask.unsqueeze(1)
    # if len(y) == 3:
    #     y = y * mask.unsqueeze(1)

    # F^H
    y = torch.fft.ifftshift(y, [-2, -1])
    x = torch.fft.ifft2(y, norm='ortho')
    x = torch.fft.fftshift(x, [-2, -1])

    # smps^H
    # print(f"[ftran] 1 x.shape: {x.shape}")
    x = x * torch.conj(smps)
    # print(f"[ftran 2] x.shape: {x.shape}")
    x = x.sum(1)
    # print(f"[ftran 3] x.shape: {x.shape}")

    return x

def ftran_non_mask(y, smps):
    """
    compute adjoint of fast MRI, x = smps^H F^H mask^H x

    :param y: under-sampled measurements, shape: batch, coils, width, height; dtype: complex
    :param smps: sensitivity maps, shape: batch, coils, width, height; dtype: complex
    :param mask: sampling mask, shape: batch, width, height; dtype: float/bool
    :return: zero-filled image
    """

    # mask^H
    # if len(y) == 2:
    #     y = y * mask.unsqueeze(0)
    # if len(y) == 3:
    #     y = y * mask.unsqueeze(1)
    # print(f"y.shape:{y.shape}")
    if (len(y.shape) == 5) and (y.shape[1] == 2):
        y = y.permute([0, 4, 2, 3, 1]).contiguous()
        y = th.view_as_complex(y)

    # F^H
    # print(f"y.shape:{y.shape}")
    y = torch.fft.ifftshift(y, [-2, -1])
    x = torch.fft.ifft2(y, norm='ortho')
    x = torch.fft.fftshift(x, [-2, -1])

    # smps^H
    
    # print(f"fMRI x 1.shape:{x.shape}")
    # print(f"fMRI smps 1.shape:{smps.shape}")
    
    x = x * torch.conj(smps)
    # print(f"x 2.shape:{x.shape}")
    if x.shape[0] == 20:
        x = x.sum(0)
        x = x.unsqueeze(0)
    elif x.shape[1] == 20:
        x = x.sum(1)
    else:
        raise ValueError(f"check again")
    # print(f"x 3.shape:{x.shape}")
        
    return x

def fmult_non_mask(x, smps):
    """
    compute forward of fast MRI, y = mask F smps x

    :param x: groundtruth or estimated image, shape: batch, width, height; dtype: complex
    :param smps: sensitivity maps, shape: batch, coils, width, height; dtype: complex
    :param mask: sampling mask, shape: batch, width, height; dtype: float/bool
    :return: undersampled measurement
    """

    # smps
    # if len(x) == 2:
    #     x = x.unsqueeze(0)
    # if len(x) == 3:
    #     x = x.unsqueeze(1)

    y = x * smps

    # F
    y = torch.fft.ifftshift(y, [-2, -1])
    y = torch.fft.fft2(y, norm='ortho')
    y = torch.fft.fftshift(y, [-2, -1])

    return y

def fmult(x, smps, mask):
    """
    compute forward of fast MRI, y = mask F smps x

    :param x: groundtruth or estimated image, shape: batch, width, height; dtype: complex
    :param smps: sensitivity maps, shape: batch, coils, width, height; dtype: complex
    :param mask: sampling mask, shape: batch, width, height; dtype: float/bool
    :return: undersampled measurement
    """

    # smps
    # print(f"[fmult] mask.shape: {mask.shape}")
    # print(f"(fmult)[len(x): {len(x)}] x.shape: {x.shape}\nmask.shape: {mask.shape}")
    # if len(x) == 2:
    #     x = x.unsqueeze(0)
    # if len(x) == 3:
    #     x = x.unsqueeze(1)
        
    # print(f"fmult fastMRI.py x.shape: {x.shape}")
    # print(f"fmult fastMRI.py mask.shape: {mask.shape}")
    # print(f"fmult fastMRI.py smps.shape: {smps.shape}")
    # print(f"fastMRI.py x.shape: {x.shape}")
    
    # The case that the input is the measurement
    if (len(x.shape) == 5):
        y = x.permute([0, 4, 2, 3, 1]).contiguous()
        y = torch.view_as_complex(y)

        # mask
        mask = mask.unsqueeze(1)
        
        # print(f"fmult fastMRI.py y.shape: {y.shape}")
        # print(f"fmult fastMRI.py mask.shape: {mask.shape}")
        
        # print(f"fastMRI.py after ifft mask.shape: {mask.shape}")
        y = y * mask
        
        y = torch.view_as_real(y).permute([0, 4, 2, 3, 1]).contiguous()

        return y

    else:
        y = x * smps
        # print(f"fastMRI.py y.shape: {y.shape}")

        # F
        y = torch.fft.ifftshift(y, [-2, -1])
        y = torch.fft.fft2(y, norm='ortho')
        y = torch.fft.fftshift(y, [-2, -1])
        
        # print(f"fastMRI.py after ifft y.shape: {y.shape}")

        # mask
        mask = mask.unsqueeze(1)
        
        # print(f"fastMRI.py after ifft mask.shape: {mask.shape}")
        y = y * mask

    return y

def uniformly_cartesian_mask(img_size, acceleration_rate, acs_percentage: float = 0.2, randomly_return: bool = False):
    if acceleration_rate == 0:
        return np.ones(img_size, dtype=np.float32)
    else:
        ny = img_size[-1]

        ACS_START_INDEX = (ny // 2) - (int(ny * acs_percentage * (2 / acceleration_rate)) // 2)
        ACS_END_INDEX = (ny // 2) + (int(ny * acs_percentage * (2 / acceleration_rate)) // 2)

        if ny % 2 == 0:
            ACS_END_INDEX -= 1

        mask = np.zeros(shape=(acceleration_rate,) + img_size, dtype=np.float32)
        
        # Generate a random starting offset
        random_offset = np.random.randint(0, acceleration_rate)
        
        mask[..., ACS_START_INDEX: (ACS_END_INDEX + 1)] = 1

        for i in range(ny):
            for j in range(acceleration_rate):
                if (i + random_offset) % acceleration_rate == j:
                    mask[j, ..., i] = 1

        if randomly_return:
            mask = mask[np.random.randint(0, acceleration_rate)]
        else:
            mask = mask[0]

        return mask


def uniformly_cartesian_mask_old(img_size, acceleration_rate, acs_percentage: float = 0.2, randomly_return: bool = False):

    ny = img_size[-1]

    ACS_START_INDEX = (ny // 2) - (int(ny * acs_percentage * (2 / acceleration_rate)) // 2)
    ACS_END_INDEX = (ny // 2) + (int(ny * acs_percentage * (2 / acceleration_rate)) // 2)

    if ny % 2 == 0:
        ACS_END_INDEX -= 1

    mask = np.zeros(shape=(acceleration_rate,) + img_size, dtype=np.float32)
    mask[..., ACS_START_INDEX: (ACS_END_INDEX + 1)] = 1

    for i in range(ny):
        for j in range(acceleration_rate):
            if i % acceleration_rate == j:
                mask[j, ..., i] = 1

    if randomly_return:
        mask = mask[np.random.randint(0, acceleration_rate)]
    else:
        mask = mask[0]

    return mask

def randomly_cartesian_mask(img_size, acceleration_rate, acs_percentage: float = 0.2, randomly_return: bool = False):
    probability = 1/acceleration_rate
    
    ny = img_size[-1]

    ACS_START_INDEX = (ny // 2) - (int(ny * acs_percentage * (2 / acceleration_rate)) // 2)
    ACS_END_INDEX = (ny // 2) + (int(ny * acs_percentage * (2 / acceleration_rate)) // 2)

    # print(ACS_END_INDEX - ACS_START_INDEX)

    if ny % 2 == 0:
        ACS_END_INDEX -= 1

    mask = np.zeros(shape=(acceleration_rate,) + img_size, dtype=np.float32)
    mask[..., ACS_START_INDEX: (ACS_END_INDEX + 1)] = 1

    for i in range(ny):
        if np.random.rand() < probability:
            mask[0, ..., i] = 1

    mask = mask[0]

    return mask

def mix_cartesian_mask(img_size, acceleration_rate, acs_percentage: float = 0.2, probability_randomly: float = 0.9):
    """
    Generate a mixed Cartesian mask with 90% probability of being randomly Cartesian and 10% of being uniform Cartesian.
    
    Parameters:
        img_size (tuple): Size of the image.
        acceleration_rate (int): Acceleration rate.
        acs_percentage (float): ACS region percentage. Default is 0.2.
        probability_randomly (float): Probability of using randomly Cartesian mask. Default is 0.9.
        
    Returns:
        np.ndarray: The generated mask.
    """
    if np.random.rand() < probability_randomly:
        # Generate a randomly Cartesian mask
        # print(f"random!!")
        return randomly_cartesian_mask(img_size, acceleration_rate, acs_percentage)
    else:
        # Generate a uniform Cartesian mask
        # print(f"uniform!!")
        return uniformly_cartesian_mask(img_size, acceleration_rate, acs_percentage)


_mask_fn = {
    'uniformly_cartesian': uniformly_cartesian_mask,
    'randomly_cartesian': randomly_cartesian_mask,
    'mix_cartesian': mix_cartesian_mask
}


def addwgn(x: torch.Tensor, input_snr):
    noiseNorm = torch.norm(x.flatten()) * 10 ** (-input_snr / 20)

    noise = torch.randn(x.size()).to(x.device)

    noise = noise / torch.norm(noise.flatten()) * noiseNorm

    y = x + noise
    return y


def check_and_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def np_normalize_minusoneto_plusone(x):
    x -= np.amin(x)
    x /= np.amax(x)

    x = (x * 2.)-1.

    return x

def np_torch_renormalize(x):
    x = (x + 1.)/2.
    return x

def np_normalize_to_uint8(x):
    x -= np.amin(x)
    x /= np.amax(x)

    x = x * 255
    x = x.astype(np.uint8)

    return x


def load_real_dataset_handle(
        idx,
        acceleration_rate: int = 1,
        is_return_y_smps_hat: bool = False,
        mask_pattern: str = 'uniformly_cartesian',
        smps_hat_method: str = 'eps',
):

    if not smps_hat_method == 'eps':
        raise NotImplementedError('smps_hat_method can only be eps now, but found %s' % smps_hat_method)

    root_path = os.path.join(ROOT_PATH, 'real')
    check_and_mkdir(root_path)

    y_h5 = os.path.join(DATA_PATH, INDEX2FILE(idx) + '.h5')

    meas_path = os.path.join(root_path, "acceleration_rate_%d_smps_hat_method_%s" % (
        acceleration_rate, smps_hat_method))
    check_and_mkdir(meas_path)

    x_hat_path = os.path.join(meas_path, 'x_hat')
    check_and_mkdir(x_hat_path)

    x_hat_h5 = os.path.join(x_hat_path, INDEX2FILE(idx) + '.h5')

    smps_hat_path = os.path.join(meas_path, 'smps_hat')
    check_and_mkdir(smps_hat_path)

    smps_hat_h5 = os.path.join(smps_hat_path, INDEX2FILE(idx) + '.h5')

    mask_path = os.path.join(meas_path, 'mask')
    check_and_mkdir(mask_path)

    mask_h5 = os.path.join(mask_path, INDEX2FILE(idx) + '.h5')

    if not os.path.exists(x_hat_h5):

        with h5py.File(y_h5, 'r') as f:
            y = f['kspace'][:]

            # Normalize the kspace to 0-1 region
            for i in range(y.shape[0]):
                y[i] /= np.amax(np.abs(y[i]))

        if not os.path.exists(mask_h5):

            _, _, n_x, n_y = y.shape
            if acceleration_rate > 1:
                mask = _mask_fn[mask_pattern]((n_x, n_y), acceleration_rate)

            else:
                mask = np.ones(shape=(n_x, n_y), dtype=np.float32)

            mask = np.expand_dims(mask, 0)  # add batch dimension
            mask = torch.from_numpy(mask)

            with h5py.File(mask_h5, 'w') as f:
                f.create_dataset(name='mask', data=mask)

        else:

            with h5py.File(mask_h5, 'r') as f:
                mask = f['mask'][:]

        if not os.path.exists(smps_hat_h5):

            os.environ['CUPY_CACHE_DIR'] = '/tmp/cupy'
            os.environ['NUMBA_CACHE_DIR'] = '/tmp/numba'
            from sigpy.mri.app import EspiritCalib
            from sigpy import Device
            import cupy

            num_slice = y.shape[0]
            iter_ = tqdm.tqdm(range(num_slice), desc='[%d, %s] Generating coil sensitivity map (smps_hat)' % (
                idx, INDEX2FILE(idx)))

            smps_hat = np.zeros_like(y)
            for i in iter_:
                tmp = EspiritCalib(y[i] * mask, device=Device(0), show_pbar=False).run()
                #tmp = cupy.asnumpy(tmp)
                smps_hat[i] = tmp

            with h5py.File(smps_hat_h5, 'w') as f:
                f.create_dataset(name='smps_hat', data=smps_hat)

            tmp = np.ones(shape=smps_hat.shape, dtype=np.uint8)
            for i in range(tmp.shape[0]):
                for j in range(tmp.shape[1]):
                    tmp[i, j] = np_normalize_to_uint8(abs(smps_hat[i, j]))
            tifffile.imwrite(smps_hat_h5.replace('.h5', '_qc.tiff'), data=tmp, compression='zlib', imagej=True)

        else:

            with h5py.File(smps_hat_h5, 'r') as f:
                smps_hat = f['smps_hat'][:]

        y = torch.from_numpy(y)
        smps_hat = torch.from_numpy(smps_hat)

        x_hat = ftran(y, smps_hat, mask)

        with h5py.File(x_hat_h5, 'w') as f:
            f.create_dataset(name='x_hat', data=x_hat)

        tmp = np.ones(shape=x_hat.shape, dtype=np.uint8)
        for i in range(x_hat.shape[0]):
            tmp[i] = np_normalize_to_uint8(abs(x_hat[i]).numpy())
        tifffile.imwrite(x_hat_h5.replace('.h5', '_qc.tiff'), data=tmp, compression ='zlib', imagej=True)

    ret = {
        'x_hat': x_hat_h5
    }
    if is_return_y_smps_hat:
        ret.update({
            'smps_hat': smps_hat_h5,
            'y': y_h5,
            'mask': mask_h5
        })

    return ret


class fastMRI(Dataset):
    def __init__(
            self,
            mode,
            acceleration_rate,
            mask_pattern,
            image_size,
            is_return_y_smps_hat: bool = True,
            smps_hat_method: str = 'eps',
            diffusion_model_type="uncond"
    ):
        if mode == 'train':
            idx_list = range(1355)
            # idx_list=range(564, 1355)
        elif mode == 'validation':
            raise ValueError("Not intended condition.")
        elif mode == 'test':
            idx_list = range(1355, 1377)
            
            
        self.diffusion_model_type = diffusion_model_type
        self.mask_pattern = mask_pattern
        self.__index_maps = []
        self.image_size = image_size

        for idx in idx_list:
            if INDEX2DROP(idx):
                # print("Found idx=[%d] annotated as DROP" % idx)
                continue

            ret = load_real_dataset_handle(
                idx,
                1,
                is_return_y_smps_hat,
                mask_pattern,
                smps_hat_method
            )

            with h5py.File(ret['x_hat'], 'r') as f:
                num_slice = f['x_hat'].shape[0]

            if INDEX2SLICE_START(idx) is not None:
                slice_start = INDEX2SLICE_START(idx)
            else:
                slice_start = 0

            if INDEX2SLICE_END(idx) is not None:
                slice_end = INDEX2SLICE_END(idx)
            else:
                slice_end = num_slice - 5

            for s in range(slice_start, slice_end):
                self.__index_maps.append([ret, s])

            self.acceleration_rate = acceleration_rate

        self.is_return_y_smps_hat = is_return_y_smps_hat

    def __len__(self):
        return len(self.__index_maps)

    def __getitem__(self, item):

        ret, s = self.__index_maps[item]

        with h5py.File(ret['x_hat'], 'r', swmr=True) as f:
            x = f['x_hat'][s]

        # if self.is_return_y_smps_hat:
        with h5py.File(ret['smps_hat'], 'r', swmr=True) as f:
            smps_hat = f['smps_hat'][s]

        with h5py.File(ret['y'], 'r', swmr=True) as f:
            y = f['kspace'][s]
            # Normalize the kspace to 0-1 region
            y /= np.amax(np.abs(y))
            # y, _ = normalize_complex(y)

        # with h5py.File(ret['mask'], 'r', swmr=True) as f:
        #     mask = f['mask'][0]
        # x = (x - np.amin(x)) / (np.amax(x) - np.amin(x))
        # * HERE IS THE CODE for normalization to ensure range of x is between [-1, 1]
        
        # x = torch.from_numpy(x)
        
        # x = torch.view_as_complex(x)#.permute([0, 3, 1, 2]).contiguous()
        y = torch.from_numpy(y)
        smps_hat = torch.from_numpy(smps_hat)
        # print(f"x.shape: {x.shape}")
        
        x_max = np.max(np.abs(x))
        x = x / x_max
        
        x = torch.from_numpy(x)
        # print(f"before normalize x.shape: {x.shape}")
        # _,n_x,n_y = y.shape
        x = (crop_images(x.unsqueeze(0).unsqueeze(0), crop_width=self.image_size)).squeeze(0).squeeze(0)

        # x = torch.view_as_real(x)#.permute([0, 3, 1, 2]).contiguous()
        # x = x.numpy()
        # print(f"before normalize x.shape: {x.shape}")
        # x = np_normalize_minusoneto_plusone(x)
        # x = torch.from_numpy(x)
        
        # x = torch.view_as_complex(x)#.permute([0, 3, 1, 2]).contiguous()
        # print(f'x.shape: {x.shape}')
        smps_hat = (crop_images(smps_hat.unsqueeze(0), crop_width=self.image_size)).squeeze(0)
        
        n_x,n_y = x.shape
        mask = _mask_fn[self.mask_pattern](img_size = (n_x, n_y), acceleration_rate = self.acceleration_rate)

        # def uniformly_cartesian_mask(img_size, acceleration_rate, acs_percentage: float = 0.2, randomly_return: bool = False):
        mask = np.expand_dims(mask, 0)  # add batch dimension
        mask = torch.from_numpy(mask).to(torch.float32)
        


        # fig, axes = plt.subplots(1, 1, figsize=(5, 5))
        # mask[0][0] = 0.
        # mask_for_plot = mask.squeeze().detach().cpu().numpy()
        # print(f"mask_for_plot: {mask_for_plot}")
        # axes.imshow(mask_for_plot, cmap='gray')
        # axes.set_title('Mask')
        # axes.axis('off')
        # plt.savefig(os.path.join(".", "mask.png"))
        # plt.close(fig)
        # raise ValueError()
        

        
        # print(f'max(mask): {torch.max(mask)}\nmin(mask): {torch.min(mask)}\nmean(mask): {torch.mean(mask)}')
        # print(f"x 1.shape:{x.shape}")
        y_hat = fmult(x, smps_hat, mask)
        x_hat = ftran(y_hat, smps_hat, mask)
        x = x.unsqueeze(0)
        smps_hat = smps_hat.unsqueeze(0)
        
        # y_hat = torch.view_as_real(y_hat)
        # B, C, H, W
        x = torch.view_as_real(x).permute([0, 3, 1, 2]).contiguous().squeeze(0)
        x_hat = torch.view_as_real(x_hat).permute([0, 3, 1, 2]).contiguous().squeeze(0)
        y_hat = torch.view_as_real(y_hat).permute([0, 4, 2, 3, 1]).contiguous()
        smps_hat = (smps_hat.permute([0, 2, 3, 1]).contiguous()).unsqueeze(1)
        # raise ValueError(f"y_hat.shape: {y_hat.shape}\nsmps.shape: {smps_hat.shape}")
        mask = mask.squeeze(0)
        
        # mask_min = torch.min(mask)
        # mask_max = torch.max(mask)
        # mask = 2 * (mask - mask_min) / (mask_max - mask_min) - 1
        
        # raise ValueError(f"x.shape:{x.shape}\nx_hat.shape:{x_hat.shape}\nsmps_hat.shape:{smps_hat.shape}\ny_hat.shape:{y_hat.shape}")
        # print(f"[fastMRI.py] torch.max(x): {torch.max(x)}\n torch.min(x): {torch.min(x)} \n torch.mean(x): {torch.mean(x)}")
        # Do it for y
        # print(f"[fastMRI.py] torch.max(y_hat): {torch.max(y_hat)}\n torch.min(y_hat): {torch.min(y_hat)} \n torch.mean(y_hat): {torch.mean(y_hat)}")
        # print(f"[fastMRI.py] torch.max(x_hat): {torch.max(x_hat)}\n torch.min(x_hat): {torch.min(x_hat)} \n torch.mean(x_hat): {torch.mean(x_hat)}")
        # print(f"[fastMRI.py] torch.max(x_hat): {torch.max(x_hat)}\n torch.min(x_hat): {torch.min(x_hat)} \n torch.mean(x_hat): {torch.mean(x_hat)}")
        # Do it for mask
        # print(f"[fastMRI.py] torch.max(mask): {torch.max(mask)}\n torch.min(mask): {torch.min(mask)} \n torch.mean(mask): {torch.mean(mask)}")
        # raise ValueError(f"self.diffusion_model_type: {self.diffusion_model_type}")
        
        # raise ValueError(f"torch.max(mask): {torch.max(mask)}\ntorch.min(mask): {torch.min(mask)}\ntorch.mean(mask): {torch.mean(mask)}")
        res_dict = {}
        if self.diffusion_model_type == "unconditional":
            res_dict.update({
                    'x': x,
                    'mask': mask
                })
        elif self.diffusion_model_type == "measurement_conditional":
            res_dict.update({
                'x': x,
                'x_hat': x_hat,
                'mask': mask
            })
        elif self.diffusion_model_type == "mask_conditional":
            res_dict.update({
                'x': x,
                'y_hat': y_hat,
                'mask': mask,
                'smps': smps_hat,
            })
        else:
            raise ValueError(f"diffusion_model_type: {self.diffusion_model_type} is not supported.")
        return res_dict

def normalize_complex(data, eps=0.):
    mag = np.abs(data)
    mag_std = mag.std()
    return data / (mag_std + eps), mag_std