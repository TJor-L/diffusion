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


def form_images_dict(path="/project/cigserver5/export1/gan.weijie/dataset/IXI"):

    res = {}

    for modality in ['T1']:
        file_paths = glob(os.path.join(path, f"IXI-{modality}", f"*{modality}.nii.gz"))
        file_paths.sort()

        for file_path in tqdm.tqdm(file_paths, desc='Forming Image Dict'):
            file_path_cp = file_path.split("/")[-1]
            patient_id = file_path_cp.replace(f"-{modality}.nii.gz", "")

            if patient_id not in res:
                res[patient_id] = {}

            res[patient_id].update({
                modality: file_path
            })

    # pop_list = []
    # for patient_id in res:
    #     if len(res[patient_id].keys()) != 4:
    #         pop_list.append(patient_id)
    #
    # for pop_key in pop_list:
    #     res.pop(pop_key)

    return res

def ftran(y, mask, smps=None):
    """
    compute adjoint of fast MRI, x = smps^H F^H mask^H x

    :param y: under-sampled measurements, shape: batch, coils, width, height; dtype: complex
    :param smps: sensitivity maps, shape: batch, coils, width, height; dtype: complex
    :param mask: sampling mask, shape: batch, width, height; dtype: float/bool
    :return: zero-filled image
    """

    if smps is not None:
        mask = mask.unsqueeze(1)

    # mask^H
    y = y * mask

    # F^H
    y = torch.fft.ifftshift(y, [-2, -1])
    x = torch.fft.ifft2(y, norm='ortho')
    x = torch.fft.fftshift(x, [-2, -1])

    if smps is not None:
        # smps^H
        x = x * torch.conj(smps)
        x = x.sum(1)

    return x


def fmult(x, mask, smps=None):
    """
    compute forward of fast MRI, y = mask F smps x

    :param x: groundtruth or estimated image, shape: batch, width, height; dtype: complex
    :param smps: sensitivity maps, shape: batch, coils, width, height; dtype: complex
    :param mask: sampling mask, shape: batch, width, height; dtype: float/bool
    :return: undersampled measurement
    """

    if smps is not None:
        # smps
        x = x.unsqueeze(1)
        y = x * smps

    else:
        y = x

    # F
    y = torch.fft.ifftshift(y, [-2, -1])
    y = torch.fft.fft2(y, norm='ortho')
    y = torch.fft.fftshift(y, [-2, -1])

    # mask
    if smps is not None:
        mask = mask.unsqueeze(1)

    y = y * mask

    return y


def uniformly_cartesian_mask(img_size, acceleration_rate, acs_percentage: float = 0.2, randomly_return: bool = False):

    ny = img_size[-1]

    ACS_START_INDEX = (ny // 2) - (int(ny * acs_percentage * (2 / acceleration_rate)) // 2)
    ACS_END_INDEX = (ny // 2) + (int(ny * acs_percentage * (2 / acceleration_rate)) // 2)

    # print(ACS_END_INDEX - ACS_START_INDEX)

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

_mask_fn = {
    'uniformly_cartesian': uniformly_cartesian_mask,
    'randomly_cartesian': randomly_cartesian_mask
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


def np_normalize_to_uint8(x):
    x -= np.amin(x)
    x /= np.amax(x)

    x = x * 255
    x = x.astype(np.uint8)

    return x


def load_synthetic_dataset_handle(
        save_path='/project/cigserver4/export2/Dataset/IXI_mri',
        acceleration_rate=4,
        noise_snr=40,
        mask_pattern: str = 'uniformly_cartesian',
):

    images_path = form_images_dict()

    ret = {}

    for patient_id in tqdm.tqdm(images_path, total=len(images_path.keys()), desc="Generating Measurement"):

        x_h5 = os.path.join(save_path, f"{patient_id}_x.h5")

        if not os.path.exists(x_h5):
            img = nib.load(images_path[patient_id]['T1'])
            x = conform(
                img,
                out_shape=(256, 256, 256),
                voxel_size=(1.0, 1.0, 1.0)
            ).get_fdata().astype(np.float32)

            mask = ants.get_mask(ants.from_numpy(x), cleanup=1).numpy()
            mask = mask.sum(0).sum(0)

            x = np.transpose(x[:, :, mask > 0], [2, 1, 0])
            x = torch.from_numpy(x)
            x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))

            with h5py.File(x_h5, 'w') as f:
                f.create_dataset(data=x, name='x')

            # tmp = np.zeros(shape=x.shape, dtype=np.uint8)
            # for i in range(x.shape[0]):
            #     tmp[i] = np_normalize_to_uint8(abs(x[i]).numpy())
            # tifffile.imwrite(x_h5.replace('.h5', '_qc.tiff'), data=tmp, compression='zlib', imagej=True)

        mask_h5 = os.path.join(save_path, f"{patient_id}_mask_acc{acceleration_rate}.h5")

        if not os.path.exists(mask_h5):
            with h5py.File(x_h5, 'r') as f:
                _, n_x, n_y = f['x'].shape

            mask = _mask_fn[mask_pattern]((n_x, n_y), acceleration_rate)
            mask = np.expand_dims(mask, 0)  # add batch dimension

            with h5py.File(mask_h5, 'w') as f:
                f.create_dataset(name='mask', data=mask)

        y_h5 = os.path.join(save_path, f"{patient_id}_y_acc{acceleration_rate}_snr{noise_snr}.h5")

        if not os.path.exists(y_h5):
            with h5py.File(x_h5, 'r') as f:
                x = f['x'][:]

            with h5py.File(mask_h5, 'r') as f:
                mask = f['mask'][:]

            x, mask = [torch.from_numpy(i) for i in [x, mask]]
            
            # raise ValueError(f"x.shape: {x.shape}")

            y = fmult(x, mask)
            y = addwgn(y, noise_snr)

            x_hat = ftran(y, mask)

            with h5py.File(y_h5, 'w') as f:
                f.create_dataset(name='y', data=y)
                f.create_dataset(name='x_hat', data=x_hat)

            # tmp = np.zeros(shape=x_hat.shape, dtype=np.uint8)
            # for i in range(x.shape[0]):
            #     tmp[i] = np_normalize_to_uint8(abs(x_hat[i]).numpy())
            # tifffile.imwrite(y_h5.replace('.h5', '_qc.tiff'), data=tmp, compression='zlib', imagej=True)

        ret.update({
            patient_id: {
                'x': x_h5,
                'y': y_h5,
                'mask': mask_h5
            }
        })

    return ret


class IXIBrain(Dataset):
    EXCLUDED_SET = []

    def __init__(
            self,
            mode,
            mask_pattern,
            path='/project/cigserver4/export2/Dataset/IXI_mri',
            acceleration_rate=4,
            noise_snr=40,
            slice_percentage=0.1,
            train_cond_diffusion=True,
    ):
        assert mode in ['train', 'validation', 'test']

        self.mode = mode
        self.train_cond_diffusion = train_cond_diffusion
        self.acceleration_rate = acceleration_rate
        self.mask_pattern = mask_pattern

        self.images_collection = load_synthetic_dataset_handle(
            save_path=path,
            acceleration_rate=acceleration_rate,
            noise_snr=noise_snr
        )
        file_paths = list(self.images_collection.keys())
        file_paths.sort()

        if self.mode == 'train':
            file_paths_selected = list(["IXI%.3d" % i for i in range(0, 550)])
        elif self.mode == 'validation':
            file_paths_selected = list(["IXI%.3d" % i for i in range(550, 594)])
        else:
            file_paths_selected = list(["IXI%.3d" % i for i in range(594, 670)])

        self.index_list = []
        self.used_subject = 0
        for file_id_full in tqdm.tqdm(file_paths, desc='Loading dataset'):

            file_id = file_id_full.split('-')[0]
            if file_id in self.EXCLUDED_SET or file_id not in file_paths_selected:
                continue
            else:
                self.used_subject += 1

            with h5py.File(os.path.join(self.images_collection[file_id_full]['x']), 'r') as f:
                num_slice = f['x'].shape[0]

            for index_slice in range(int(num_slice * slice_percentage), int(num_slice * (1 - slice_percentage))):
                self.index_list.append([file_id_full, index_slice])

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, item):

        file_id_full, index_slice = self.index_list[item]

        res_dict = {
            'index_slice': index_slice,
            'file_id_full': file_id_full
        }
        
        # print(f"file_id_full: {file_id_full}")

        with h5py.File(self.images_collection[file_id_full]['x'], 'r', swmr=True) as f:
            # print(f"[ixi_brain.py] f['x']: {f['x']}")
            x = torch.from_numpy(f['x'][index_slice]).to(torch.float32).unsqueeze(0)

        if self.train_cond_diffusion == True:
            with h5py.File(self.images_collection[file_id_full]['y'], 'r', swmr=True) as f:
                # print(f"[ixi_brain.py] f['x_hat']: {f['x_hat']}")
                x_hat = torch.from_numpy(abs(f['x_hat'][index_slice])).to(torch.float32).unsqueeze(0)

            with h5py.File(self.images_collection[file_id_full]['mask'], 'r', swmr=True) as f:
                # print(f"[ixi_brain.py] f['mask']: {f['mask']}")
                # print(f"[ixi_brain.py] self.images_collection: {self.images_collection}")
                mask = torch.from_numpy(f['mask'][0]).to(torch.float32).unsqueeze(0)
            # print(f"[ixi_brain.py] x.shape: {x.shape}")
            # print(f"[ixi_brain.py] x_hat.shape: {x_hat.shape}")
            # print(f"[ixi_brain.py] mask.shape: {mask.shape}")
            # raise ValueError() 
            # print(f"x.shape: {x.shape}")
            _, n_x, n_y = x.shape
            # mask_pattern = 'uniformly_cartesian'
            mask = _mask_fn[self.mask_pattern]((n_x, n_y), self.acceleration_rate)
            mask = np.expand_dims(mask, 0)  # add batch dimension
            mask = torch.from_numpy(mask[0]).to(torch.float32).unsqueeze(0)
            
            
            ####
            # Output format is x.shape: torch.Size([1, 256, 256]) / mask.shape: torch.Size([1, 256, 256])
            ####
            # print(f"[ixi_brain.py] x.shape: {x.shape} / mask.shape: {mask.shape}")
            # print(f"[ixi_brain.py] torch.max(x): {torch.max(x)}\n torch.min(x): {torch.min(x)} \n torch.mean(x): {torch.mean(x)}")
            # print(f"[ixi_brain.py] torch.max(x_hat): {torch.max(x_hat)}\n torch.min(x_hat): {torch.min(x_hat)} \n torch.mean(x_hat): {torch.mean(x_hat)}")
            # # x = 2*x -1
            # x_hat = 2*x_hat -1
            
            # print(f"[ixi_brain.py] x.shape: {x.shape} / mask.shape: {mask.shape}")
            # print(f"[ixi_brain.py] torch.max(x): {torch.max(x)}\n torch.min(x): {torch.min(x)} \n torch.mean(x): {torch.mean(x)}")
            # print(f"[ixi_brain.py] torch.max(x_hat): {torch.max(x_hat)}\n torch.min(x_hat): {torch.min(x_hat)} \n torch.mean(x_hat): {torch.mean(x_hat)}")
            

            res_dict.update({
                'x': x,
                'x_hat': x_hat,
                'mask': mask
            })
        else:
            res_dict.update({
                'x': x,
                'mask': mask
            })

        # with h5py.File(os.path.join(file_path), 'r', swmr=True) as f:
        #     for modality in ['T1']:
        #         res_dict.update({
        #             modality: torch.from_numpy(f[modality][index_slice]).to(torch.float32).unsqueeze(0)
        #         })

        return res_dict


if __name__ == '__main__':
    # dataset = IXIBrain(mode='tra', acceleration_rate=10)
    # print(len(dataset))
    # res = dataset[100]
    # for k in ['x', 'x_hat']:
    #     print(k, res[k].shape, res[k].dtype)

    # generate_dataset(save_path='/project/cigserver4/export2/Dataset/IXI_registered')
    # IXIBrain('tra', path='/project/cigserver4/export2/Dataset/IXI_registered')

    print(uniformly_cartesian_mask((256, 256), 30).sum() / (256 * 256))
