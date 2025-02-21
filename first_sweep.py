import numpy as np
import tifffile
import torch

from utility.utility import copy_code_to_path, get_save_path_from_config, check_and_mkdir
from utility.tweedie_utility import mkdir_exp_recording_folder
# import pytorch_lightning as lightning

from mpi4py import MPI
import os
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import datetime

import argparse
import multiprocessing
import yaml
import sys
from second_main_experiment import train, test

####
""" # *Data shape discussion
• Shape of network input: (b,c,w,h)
    IXI_Brain: (1,1,256,256)
    fastMRI: ()
• Shape of network output
    UNet forward
        x.shape: torch.Size([1, 2, 320, 320])
    SuperUNet forward
        x.shape: torch.Size([1, 2, 320, 320, 20]) 
        smps.shape: torch.Size([1, 1, 320, 320, 20]) 
        low_res.shape: torch.Size([1, 1, 320, 320])

• Shape of abs_helper input

• Shape of crop_image input

• Shape of fmult input
    x.shape: torch.Size([320, 320])
    smps_hat.shape: torch.Size([20, 320, 320])
    mask.shape: torch.Size([1, 320, 320])

• Shape of ftran(y_hat, smps_hat, mask) input
    y_hat.shape: torch.Size([1, 20, 320, 320])
    smps_hat.shape: torch.Size([20, 320, 320])
    mask.shape: torch.Size([1, 320, 320])
    
• Shape of q_sample(x_start, t, noise=noise) input
    x_start.shape: torch.Size([1, 2, 320, 320])

• Shape of torch.view_as_complex input:

• Value range of fastMRI ground truth is range of [-1, 1]

• Mask multiplication
    fastMRI.py after ifft y.shape: torch.Size([20, 320, 320])
    fastMRI.py after ifft mask.shape: torch.Size([1, 1, 320, 320])
    
• Shape for visualization:
    
"""
####


def main():
    multiprocessing.set_start_method(
        "spawn"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--task_config', type=str)
    args = parser.parse_args()
    
    gpu = args.gpu
    device_str = f"cuda:{gpu}" if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)  

    ###############
    # Load configurations
    ###############
    task_config = load_yaml(args.task_config)
    ###############
    # Extract information from configurations
    ###############
    save_dir = task_config['setting']['save_dir']
    save_code = task_config['setting']['save_code']
    code_dir = task_config['setting']['code_dir']
    experiment_type = task_config['setting']['experiment_type']
    diffusion_noise_schedule = task_config['vp_diffusion']['noise_schedule']
    diffusion_model_type  = task_config['vp_diffusion']['diffusion_model_type']
    image_size = task_config['vp_diffusion']['image_size']
    microbatch = task_config['vp_diffusion']['microbatch']
    learn_sigma = task_config['vp_diffusion']['learn_sigma']
    in_channels = task_config['vp_diffusion']['in_channels']
    cond_channels = task_config['vp_diffusion']['cond_channels']
    dataset_name = task_config['dataset']['dataset_name'].lower()
    mask_pattern = task_config['dataset']['mask_pattern'].lower()
    acceleration_rate = task_config['dataset']['acceleration_rate']
    num_multi_slice = task_config['dataset']['num_multi_slice']
    noise_snr = task_config['dataset']['noise_snr']
    batch_size = task_config['train']['batch_size']
    compute_vb_loss = task_config['train']['compute_vb_loss']
    num_workers = task_config['train']['num_workers']
    resume_checkpoint = task_config['setting']['resume_checkpoint']
    save_interval = task_config['setting']['save_interval']
    pretrained_check_point = task_config['vp_diffusion']['model_path']
    
    num_iters = task_config['langevin']['num_iters']
    
    num_channels = task_config['vp_diffusion']['num_channels']
    num_res_blocks = task_config['vp_diffusion']['num_res_blocks']
    channel_mult = task_config['vp_diffusion']['channel_mult']
    class_cond = task_config['vp_diffusion']['class_cond']
    use_checkpoint = task_config['vp_diffusion']['use_checkpoint']
    attention_resolutions = task_config['vp_diffusion']['attention_resolutions']
    num_heads = task_config['vp_diffusion']['num_heads']
    num_head_channels = task_config['vp_diffusion']['num_head_channels']
    num_heads_upsample = task_config['vp_diffusion']['num_heads_upsample']
    use_scale_shift_norm = task_config['vp_diffusion']['use_scale_shift_norm']
    dropout = task_config['vp_diffusion']['dropout']
    resblock_updown = task_config['vp_diffusion']['resblock_updown']
    use_fp16 = task_config['vp_diffusion']['use_fp16']
    use_new_attention_order = task_config['vp_diffusion']['use_new_attention_order']

    
    use_ddim = task_config['vp_diffusion']['use_ddim']
    timestep_respacing = task_config['vp_diffusion']['timestep_respacing']
    clip_denoised = task_config['vp_diffusion']['clip_denoised']
    
    # if task_config['setting']['solve_inverse_problem'] == True:
    #     inverse_problem_config = task_config['measurement']
    # else:
    inverse_problem_config = None
    
    diffusion_model_config_dict = {
        'noise_schedule': task_config['vp_diffusion']['noise_schedule'],
        'image_size': task_config['vp_diffusion']['image_size'],
        'large_size': task_config['vp_diffusion']['image_size'],
        'small_size': task_config['vp_diffusion']['image_size'],
        'batch_size': task_config['train']['batch_size'],
        'microbatch': task_config['vp_diffusion']['microbatch'],
        'learn_sigma': task_config['vp_diffusion']['learn_sigma'],
        'in_channels': task_config['vp_diffusion']['in_channels'],
        'cond_channels': task_config['vp_diffusion']['cond_channels'],
        'num_channels': task_config['vp_diffusion']['num_channels'],
        'num_res_blocks': task_config['vp_diffusion']['num_res_blocks'],
        'channel_mult': task_config['vp_diffusion']['channel_mult'],
        'use_checkpoint': task_config['vp_diffusion']['use_checkpoint'],
        'attention_resolutions': task_config['vp_diffusion']['attention_resolutions'],
        'num_heads': task_config['vp_diffusion']['num_heads'],
        'num_head_channels': task_config['vp_diffusion']['num_head_channels'],
        'num_heads_upsample': task_config['vp_diffusion']['num_heads_upsample'],
        'use_scale_shift_norm': task_config['vp_diffusion']['use_scale_shift_norm'],
        'dropout': task_config['vp_diffusion']['dropout'],
        'resblock_updown': task_config['vp_diffusion']['resblock_updown'],
        'use_fp16': task_config['vp_diffusion']['use_fp16'],
        'use_new_attention_order': task_config['vp_diffusion']['use_new_attention_order'],
        'timestep_respacing': task_config['vp_diffusion']['timestep_respacing'],
        'model_path': task_config['vp_diffusion']['model_path'],
        'use_ddim': task_config['vp_diffusion']['use_ddim']
    }
    
    METHOD_LIST = {
        'train': train,
        'test': test,
    }


    METHOD_LIST[experiment_type](
        save_dir=save_dir,
        save_code=save_code,
        code_dir=code_dir,
        experiment_type=experiment_type,
        diffusion_noise_schedule=diffusion_noise_schedule,
        diffusion_model_type=diffusion_model_type,
        dataset_name=dataset_name,
        mask_pattern=mask_pattern,
        acceleration_rate=acceleration_rate,
        num_multi_slice=num_multi_slice,
        noise_snr=noise_snr,
        batch_size=batch_size,
        num_workers=num_workers,
        resume_checkpoint = resume_checkpoint,
        save_interval=save_interval,
        image_size=image_size,
        microbatch=microbatch,
        learn_sigma=learn_sigma,
        in_channels=in_channels,
        cond_channels=cond_channels,
        num_channels=num_channels,
        num_res_blocks=num_res_blocks,
        channel_mult=channel_mult,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        use_new_attention_order=use_new_attention_order,
        device=device,
        gpu=gpu,
        pretrained_check_point=pretrained_check_point,
        use_ddim=use_ddim,
        timestep_respacing=timestep_respacing,
        clip_denoised=clip_denoised,
        diffusion_model_config_dict = diffusion_model_config_dict,
        num_iters = num_iters,
        compute_vb_loss = compute_vb_loss,
        inverse_problem_config = inverse_problem_config
    )


"""
class DatasetWrapper(Dataset):
    def __init__(self, parent_dataset, config, diffusion_model_type):
        self.dataset = parent_dataset
        self.diffusion_model_type = diffusion_model_type

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        # print(f"item: {item}")
        if self.diffusion_model_type in ["measurement_conditional", "mask_conditional"]:
            x = self.dataset[item]['x']
            x_hat = self.dataset[item]['x_hat']
            mask = self.dataset[item]['mask']
            return x, {'low_res': x_hat, 'mask': mask}
        else:
            x = self.dataset[item]['x']
            # x_hat = self.dataset[item]['x_hat']
            return x, {}#, {'low_res': x_hat}

def training_dataloader_wrapper(dataset, config):
    data_loader = DataLoader(
        dataset=dataset,
        shuffle=True,
        batch_size=config['train']['batch_size'],
        num_workers=config['train']['num_worker'],
    )

    while True:
        yield from data_loader


def sampling_dataloader_wrapper(dataset, config):
    data_loader = DataLoader(
        dataset=dataset,
        shuffle=False,
        batch_size=1,
        num_workers=0,
    )

    while True:
        yield from data_loader

"""
def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

# def run(config):
#     if config['setting']['mode'] == 'tra':
#         train(config)

#     elif config['setting']['mode'] == 'tst':
#         test(config)


if __name__ == "__main__":
    main()