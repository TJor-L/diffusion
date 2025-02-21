import numpy as np
import tifffile
import torch

from .guided_diffusion import dist_util, logger
from .scripts.image_train import create_argparser as create_argparser_uncond
from .scripts.super_res_train import create_argparser as create_argparser_cond
from .scripts.super_res_sample import create_argparser as create_argparser_sampling
from .guided_diffusion.resample import create_named_schedule_sampler
from .guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser
)
from .guided_diffusion.train_util import TrainLoop
from utility.utility import copy_code_to_path, get_save_path_from_config, check_and_mkdir
from utility.tweedie_utility import mkdir_exp_recording_folder
# from utility.logger import get_logger
# import pytorch_lightning as lightning

from mpi4py import MPI
import os
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import datetime


import argparse

class DatasetWrapper(Dataset):
    def __init__(self, parent_dataset, config, train_cond_diffusion):
        self.dataset = parent_dataset
        self.train_cond_diffusion = train_cond_diffusion

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        if self.train_cond_diffusion == True:
            x = self.dataset[item]['x']
            x_hat = self.dataset[item]['x_hat']
            return x, {'low_res': x_hat}
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


def run(config):
    if config['setting']['mode'] == 'tra':
        train(config)

    elif config['setting']['mode'] == 'tst':
        test(config)

import time

def train(config):
    
    # def mkdir_exp_recording_folder(save_dir, acc_rate, dataset_name):
    # save_dir = config['setting']['save_dir'] + time +
    save_dir, _ = mkdir_exp_recording_folder(save_dir=config['setting']['save_dir'], acc_rate=config['dataset']['acceleration_rate'], dataset_name="brain")
    # raise ValueError(f"save_dir: {save_dir}")
    
    # raise ValueError(f"get_save_path_from_config(config): {get_save_path_from_config(config)}")
    if config['setting']['save_code'] == True:
        if MPI.COMM_WORLD.Get_rank() == 0:
            copy_code_to_path(src_path=config['setting']['code_dir'], file_path=get_save_path_from_config(config))
    else:
        check_and_mkdir(config['setting']['save_dir'])


    # os.environ['OPENAI_LOGDIR'] = get_save_path_from_config(config)  # set the logdir
    # train_cond_diffusion = True
    train_cond_diffusion = False

    if train_cond_diffusion == True:
        args = create_argparser_cond().parse_args()
    else:
        args = create_argparser_uncond().parse_args()
    raise ValueError(f"args: {args}")
    # args_dict = dict(
    #     data_dir="",
    #     schedule_sampler="uniform",
    #     lr=0.0001,
    #     weight_decay=0.0,
    #     lr_anneal_steps=0,
    #     batch_size=1,
    #     microbatch=-1,
    #     ema_rate="0.9999",
    #     log_interval=10,
    #     save_interval=10000,
    #     resume_checkpoint="",
    #     use_fp16=False,
    #     fp16_scale_growth=0.001,
    #     num_channels=128,
    #     num_res_blocks=2,
    #     num_heads=4,
    #     num_heads_upsample=-1,
    #     num_head_channels=-1,
    #     attention_resolutions="16,8",
    #     dropout=0.0,
    #     class_cond=False,
    #     use_checkpoint=False,
    #     use_scale_shift_norm=True,
    #     resblock_updown=False,
    #     in_channels=3,
    #     learn_sigma=False,
    #     diffusion_steps=1000,
    #     noise_schedule="linear",
    #     timestep_respacing="",
    #     use_kl=False,
    #     predict_xstart=False,
    #     rescale_timesteps=False,
    #     rescale_learned_sigmas=False,
    #     large_size=256,
    #     small_size=64
    # )
    # args = argparse.ArgumentParser()
    # add_dict_to_argparser(args, args_dict)
    
    # raise ValueError(f"args: {args}")

    args.save_interval = 10
    args.large_size = 256
    args.small_size = 256
    args.image_size = 256
    args.batch_size = config['train']['batch_size']
    args.microbatch = -1
    args.learn_sigma = True
    args.noise_schedule = config['vp_diffusion']['noise_schedule']
    args.in_channels = 1
    
    # print(f"args: {args}")

    dist_util.setup_dist(config)

    logger = get_logger()
    
    gpu = config['setting']['gpu_index']

    device_str = f"cuda:{gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    
    # logger.configure()

    logger.info(f"creating model and diffusion...")
    # logger.log("creating model and diffusion...")
    
    
    if train_cond_diffusion == True:
        # raise ValueError(f"args:{args}\nsr_model_and_diffusion_defaults().keys(): {sr_model_and_diffusion_defaults(train_cond_diffusion=train_cond_diffusion).keys()}")
        model, diffusion = sr_create_model_and_diffusion(
            **args_to_dict(args, sr_model_and_diffusion_defaults(train_cond_diffusion=train_cond_diffusion).keys())
        )
        
    else:
        # raise ValueError(f"args: {args}\nmodel_and_diffusion_defaults().keys(): {model_and_diffusion_defaults().keys()}")
        # raise ValueError(f"args_to_dict(args, model_and_diffusion_defaults(train_cond_diffusion=train_cond_diffusion).keys()): {args_to_dict(args, model_and_diffusion_defaults(train_cond_diffusion=train_cond_diffusion).keys())}")
        # raise ValueError(f"args_to_dict(args, model_and_diffusion_defaults(train_cond_diffusion=train_cond_diffusion).keys()): {args_to_dict(args, model_and_diffusion_defaults(train_cond_diffusion=train_cond_diffusion).keys())}")
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults(train_cond_diffusion=train_cond_diffusion).keys())
        )
        
    # raise ValueError(f"args: {args}\nsr_model_and_diffusion_defaults():{sr_model_and_diffusion_defaults()}\nsr_model_and_diffusion_defaults().keys():{sr_model_and_diffusion_defaults().keys()}")
    
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    

    logger.info(f"creating data loader...")
    # logger.log("creating data loader...")
    from dataset.ixi_brain import IXIBrain
    dataset = IXIBrain(mode='tra', acceleration_rate=config['dataset']['acceleration_rate'])
    dataset = DatasetWrapper(dataset, config, train_cond_diffusion=train_cond_diffusion)
    data = training_dataloader_wrapper(dataset, config)

    # logger.log("training...")
    logger.info(f"training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        save_dir = save_dir
    ).run_loop()


def test(config):

    save_path = os.path.join(
        get_save_path_from_config(config),
        "TEST[%s]_%s_%s_ckpt_%s" % (
            datetime.datetime.now().strftime("%m%d%H%M"),
            config['setting']['exp_folder'],
            config['test']['ckpt_path'],
            config['test']['desc'],
        ),
    )
    check_and_mkdir(save_path)

    if MPI.COMM_WORLD.Get_rank() == 0:
        copy_code_to_path(src_path=config['setting']['src_path'], file_path=save_path)

    # os.environ['OPENAI_LOGDIR'] = get_save_path_from_config(config)  # set the logdir
    # lightning.seed_everything(2110)

    args = create_argparser_sampling().parse_args()
    args.large_size = 256
    args.small_size = 256
    args.image_size = 256
    args.model_path = os.path.join(get_save_path_from_config(config), config['test']['ckpt_path'])
    args.batch_size = 1
    args.use_ddim = config['test']['diffusion']['use_ddim']
    args.timestep_respacing = config['test']['diffusion']['timestep_respacing']
    args.learn_sigma = True
    args.clip_denoised = True
    # ! HERE change cosine as arg.input
    args.noise_schedule = 'cosine'
    args.in_channels = 1

    dist_util.setup_dist(config)
    raise ValueError("logging has to be modified")
    logger.configure()

    logger.log("creating model...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading data...")
    from dataset.ixi_brain import IXIBrain
    dataset = IXIBrain(mode='tst', acceleration_rate=config['dataset']['acceleration_rate'])
    dataset = DatasetWrapper(dataset, config)
    data = sampling_dataloader_wrapper(dataset, config)

    logger.log("sampling...")
    result, low_res = [], []
    for i in range(300):

        print(f"start generating the {i}th image")

        x_gt, model_kwargs = next(data)

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 1, 256, 256),
            clip_denoised=args.clip_denoised,
            model_kwargs={'low_res': model_kwargs['low_res'].cuda()},
            progress=True,
        )
        sample = sample.detach().cpu().numpy()

        result.append(sample)
        low_res.append(model_kwargs['low_res'].numpy())

    result = np.concatenate(result, 0)
    low_res = np.concatenate(low_res, 0)

    tifffile.imwrite(
        os.path.join(save_path, 'sampled_img.tiff'), result)

    tifffile.imwrite(
        os.path.join(save_path, 'low_res.tiff'), low_res)
