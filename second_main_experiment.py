import numpy as np
import tifffile
import torch

from guided_diffusion.guided_diffusion import dist_util, logger
from guided_diffusion.scripts.image_train import create_argparser as create_argparser_uncond
from guided_diffusion.scripts.super_res_train import create_argparser as create_argparser_cond
from guided_diffusion.scripts.super_res_sample import create_argparser as create_argparser_sampling
from guided_diffusion.guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser
)
from guided_diffusion.guided_diffusion.train_util import TrainLoop
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
from utility.utility import crop_images, abs_helper
import matplotlib.pyplot as plt

from guided_diffusion.guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.guided_diffusion.measurements import get_noise, get_operator

from functools import partial

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

class DatasetWrapper(Dataset):
    def __init__(self, parent_dataset, diffusion_model_type):
        self.dataset = parent_dataset
        self.diffusion_model_type = diffusion_model_type

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        # print(f"item: {item}")
        if self.diffusion_model_type in ['measurement_conditional']:
            x = self.dataset[item]['x']
            x_hat = self.dataset[item]['x_hat']
            mask = (self.dataset[item]['mask']).unsqueeze(1)
            # return y_hat, x, {'low_res': x_hat}
            return x, x, {'low_res': x_hat}
        
        # elif self.diffusion_model_type in ['mask_conditional']:
        #     x = self.dataset[item]['x']
        #     y_hat = self.dataset[item]['y_hat'].squeeze(0)
        #     mask = (self.dataset[item]['mask']).unsqueeze(0)
        #     smps = (self.dataset[item]['smps']).squeeze(0)
        #     # raise ValueError(f"y_hat.shape: {y_hat.shape}\nmask.shape: {mask.shape}\nsmps.shape: {smps.shape}")

        #     # return x, {'low_res': mask}
        #     return y_hat, x, {'low_res': mask, 'smps': smps}
        
        else:
            x = self.dataset[item]['x']
            # x_hat = self.dataset[item]['x_hat']
            return x, x, {}#, {'low_res': x_hat}


def training_dataloader_wrapper(dataset, batch_size, num_workers):
    data_loader = DataLoader(
        dataset=dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    while True:
        yield from data_loader


def sampling_dataloader_wrapper(dataset, batch_size, num_workers):
    data_loader = DataLoader(
        dataset=dataset,
        shuffle=False,
        batch_size=1,
        num_workers=0,
    )

    while True:
        yield from data_loader

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

import time

# def train(config):
def train(
    save_dir,
    save_code,
    code_dir,
    experiment_type,
    diffusion_noise_schedule,
    diffusion_model_type,
    dataset_name,
    mask_pattern,
    acceleration_rate,
    num_multi_slice,
    noise_snr,
    batch_size,
    num_workers,
    save_interval,
    resume_checkpoint,
    image_size,
    microbatch,
    learn_sigma,
    in_channels,
    cond_channels,
    num_channels,
    num_res_blocks,
    channel_mult,
    class_cond,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
    resblock_updown,
    use_fp16,
    use_new_attention_order,
    gpu,
    device,
    pretrained_check_point,
    use_ddim,
    timestep_respacing,
    clip_denoised,
    diffusion_model_config_dict,
    num_iters,
    compute_vb_loss,
    inverse_problem_config = None
):

    save_dir, _ = mkdir_exp_recording_folder(save_dir=save_dir, acc_rate=acceleration_rate, dataset_name=dataset_name, compute_vb_loss = compute_vb_loss, mask_pattern = mask_pattern, image_size=image_size)
    if save_code == True:
        if MPI.COMM_WORLD.Get_rank() == 0:
            # copy_code_to_path(src_path=code_dir, file_path=get_save_path_from_config(config))
            copy_code_to_path(src_path=code_dir, file_path=save_dir)
    else:
        check_and_mkdir(save_dir)

    os.environ['OPENAI_LOGDIR'] = str(save_dir) # set the logdir

    if diffusion_model_type in ["measurement_conditional", "mask_conditional"]:
        args = create_argparser_cond().parse_args([])
    else:
        args = create_argparser_uncond().parse_args([])
    
    args.save_interval = save_interval
    
    args.cond_channels = cond_channels
    args.resume_checkpoint = resume_checkpoint
    
    for key, value in diffusion_model_config_dict.items():
        if hasattr(args, key):  # Ensure the attribute exists in args
            setattr(args, key, value)
        else:
            print(f"key: {key}")
    
    dist_util.setup_dist(gpu=gpu)
    logger.configure()
    
    logger.log("creating model and diffusion...")
    
    # raise ValueError(f"args_to_dict(args, sr_model_and_diffusion_defaults(diffusion_model_type=diffusion_model_type).keys()): {args_to_dict(args, sr_model_and_diffusion_defaults(diffusion_model_type=diffusion_model_type).keys())}")
    if diffusion_model_type in ["measurement_conditional", "mask_conditional"]:
        # raise ValueError(f"args: {args}\nsr_model_and_diffusion_defaults():{sr_model_and_diffusion_defaults(diffusion_model_type=diffusion_model_type)}\nsr_model_and_diffusion_defaults().keys():{sr_model_and_diffusion_defaults(diffusion_model_type=diffusion_model_type).keys()}")
        model, diffusion = sr_create_model_and_diffusion(
            **args_to_dict(args, sr_model_and_diffusion_defaults(diffusion_model_type=diffusion_model_type).keys())
        )
        
    else:
        # raise ValueError(f"args: {args}\nsr_model_and_diffusion_defaults():{sr_model_and_diffusion_defaults(diffusion_model_type=diffusion_model_type)}\nsr_model_and_diffusion_defaults().keys():{sr_model_and_diffusion_defaults(diffusion_model_type=diffusion_model_type).keys()}")
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults(diffusion_model_type=diffusion_model_type).keys())
        )
        
    # raise ValueError(f"args: {args}\nsr_model_and_diffusion_defaults():{sr_model_and_diffusion_defaults()}\nsr_model_and_diffusion_defaults().keys():{sr_model_and_diffusion_defaults().keys()}")
    
    # print(f"2 [sme.py] memory_allocated: {torch.cuda.memory_allocated(device='cuda:0') / (1024 * 1024 * 1024):2f}")

    model.to(dist_util.dev(gpu=gpu))
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    


    logger.log("creating data loader...")
    from dataset.ixi_brain import IXIBrain
    from dataset.fastMRI import fastMRI, randomly_cartesian_mask, uniformly_cartesian_mask, mix_cartesian_mask
    if dataset_name == "ixibrain":
        dataset = IXIBrain(mode='train', acceleration_rate=acceleration_rate, mask_pattern=mask_pattern)
        dataset = DatasetWrapper(dataset, diffusion_model_type=diffusion_model_type)
        data = training_dataloader_wrapper(dataset, batch_size=batch_size, num_workers=num_workers)
    elif dataset_name == "fastmri":
        dataset = fastMRI(mode='train', acceleration_rate=acceleration_rate, mask_pattern=mask_pattern, image_size = image_size, diffusion_model_type=diffusion_model_type)
        dataset = DatasetWrapper(dataset, diffusion_model_type=diffusion_model_type)
        data = training_dataloader_wrapper(dataset, batch_size=batch_size, num_workers=num_workers)
    else:
        raise ValueError("Check the dataset_name")
    
    model = model.to(device)

    # print(f"3 [sme.py] memory_allocated: {torch.cuda.memory_allocated(device='cuda:0') / (1024 * 1024 * 1024):2f}")

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        # log_interval=args.log_interval,
        log_interval=50,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        save_dir = save_dir,
        gpu = gpu,
        compute_vb_loss = compute_vb_loss,
        acceleration_rate = acceleration_rate
    ).run_loop()


# def test(config):
def test(
    save_dir,
    save_code,
    code_dir,
    experiment_type,
    diffusion_noise_schedule,
    diffusion_model_type,
    dataset_name,
    mask_pattern,
    acceleration_rate,
    num_multi_slice,
    noise_snr,
    batch_size,
    num_workers,
    resume_checkpoint,
    save_interval,
    image_size,
    microbatch,
    learn_sigma,
    in_channels,
    cond_channels,
    num_channels,
    num_res_blocks,
    channel_mult,
    class_cond,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
    resblock_updown,
    use_fp16,
    use_new_attention_order,
    gpu,
    device,
    pretrained_check_point,
    use_ddim,
    timestep_respacing,
    clip_denoised,
    diffusion_model_config_dict,
    num_iters,
    compute_vb_loss,
    inverse_problem_config = None,
):

    """
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
    """
    # save_dir, _ = mkdir_exp_recording_folder(save_dir=save_dir, acc_rate=acceleration_rate, dataset_name=dataset_name)
    save_dir, _ = mkdir_exp_recording_folder(save_dir=save_dir, acc_rate=acceleration_rate, dataset_name=dataset_name, compute_vb_loss = compute_vb_loss, mask_pattern = mask_pattern, image_size=image_size)
    
    sample_conditionally = True if inverse_problem_config is not None else False
    
    if save_code == True:
        if MPI.COMM_WORLD.Get_rank() == 0:
            # copy_code_to_path(src_path=code_dir, file_path=get_save_path_from_config(config))
            copy_code_to_path(src_path=code_dir, file_path=save_dir)
    else:
        check_and_mkdir(save_dir)
    

    # if MPI.COMM_WORLD.Get_rank() == 0:
    #     copy_code_to_path(src_path=config['setting']['src_path'], file_path=save_dir)

    # os.environ['OPENAI_LOGDIR'] = get_save_path_from_config(config)  # set the logdir
    os.environ['OPENAI_LOGDIR'] = str(save_dir) # set the logdir
    # lightning.seed_everything(2110)

    # args = create_argparser_sampling().parse_args()
    if diffusion_model_type in ["measurement_conditional", "mask_conditional"]:
        args = create_argparser_cond().parse_args([])
    else:
        args = create_argparser_uncond().parse_args([])
        
    for key, value in diffusion_model_config_dict.items():
        if hasattr(args, key):  # Ensure the attribute exists in args
            setattr(args, key, value)
        else:
            print(f"key: {key}")
    
    
    # args.save_interval = save_interval
    # args.large_size = image_size
    # args.small_size = image_size
    # args.image_size = image_size
    # args.batch_size = batch_size
    # args.microbatch = microbatch
    # args.learn_sigma = learn_sigma
    # args.noise_schedule = diffusion_noise_schedule
    # args.in_channels = in_channels
    # raise ValueError(f"diffusion_model_config_dict:{diffusion_model_config_dict}")
    # for key, value in diffusion_model_config_dict.items():
    #     if hasattr(args, key):  # Ensure the attribute exists in args
    #         setattr(args, key, value)
    #     else:
    #         print(f"key: {key}")
    
    args.model_path = os.path.join(save_dir, diffusion_model_config_dict['model_path'])
    args.use_ddim = diffusion_model_config_dict['use_ddim']
    
    args.cond_channels = cond_channels

    args.resume_checkpoint = resume_checkpoint

    # dist_util.setup_dist(config)
    dist_util.setup_dist(gpu=gpu)
    logger.configure()

    logger.log("creating model...")
    # model, diffusion = sr_create_model_and_diffusion(
    #     **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    # )

    # raise ValueError(f"args:{args}\nsr_model_and_diffusion_defaults().keys(): {sr_model_and_diffusion_defaults(train_cond_diffusion=train_cond_diffusion).keys()}")
    if diffusion_model_type in ["measurement_conditional", "mask_conditional"]:
        # raise ValueError(f"args: {args}\nsr_model_and_diffusion_defaults():{sr_model_and_diffusion_defaults(diffusion_model_type=diffusion_model_type)}\nsr_model_and_diffusion_defaults().keys():{sr_model_and_diffusion_defaults(diffusion_model_type=diffusion_model_type).keys()}")
        print("in_channels:", args.in_channels)
        print("cond_channels:", args.cond_channels)
        model, diffusion = sr_create_model_and_diffusion(
            **args_to_dict(args, sr_model_and_diffusion_defaults(diffusion_model_type=diffusion_model_type).keys())
        )
        
    else:
        # raise ValueError(f"args: {args}\nsr_model_and_diffusion_defaults():{sr_model_and_diffusion_defaults(diffusion_model_type=diffusion_model_type)}\nsr_model_and_diffusion_defaults().keys():{sr_model_and_diffusion_defaults(diffusion_model_type=diffusion_model_type).keys()}")
        # raise ValueError(f"args: {args}\nmodel_and_diffusion_defaults().keys(): {model_and_diffusion_defaults().keys()}")
        # raise ValueError(f"args_to_dict(args, model_and_diffusion_defaults(train_cond_diffusion=train_cond_diffusion).keys()): {args_to_dict(args, model_and_diffusion_defaults(train_cond_diffusion=train_cond_diffusion).keys())}")
        # raise ValueError(f"args_to_dict(args, model_and_diffusion_defaults(train_cond_diffusion=train_cond_diffusion).keys()): {args_to_dict(args, model_and_diffusion_defaults(train_cond_diffusion=train_cond_diffusion).keys())}")

        model, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults(diffusion_model_type=diffusion_model_type).keys())
        )
    
    # raise ValueError(f"args: {args_to_dict(args, model_and_diffusion_defaults(diffusion_model_type=diffusion_model_type).keys())}")
    
    # 先加载 checkpoint 到一个变量
    ckpt = dist_util.load_state_dict(args.model_path, map_location="cpu")

    # 假设模型的第一层在 input_blocks[0][0]，打印它的权重形状
    model_input_weight_shape = model.input_blocks[0][0].weight.shape
    ckpt_input_weight_shape = ckpt["input_blocks.0.0.weight"].shape

    print("模型第一层权重形状:", model_input_weight_shape)
    print("Checkpoint 中对应权重形状:", ckpt_input_weight_shape)

    # 如果只关注输入通道数，可以打印第二个维度
    print("模型输入通道数:", model_input_weight_shape[1])
    print("Checkpoint 中输入通道数:", ckpt_input_weight_shape[1])

    # 然后再调用 load_state_dict
    model.load_state_dict(ckpt)



    print(args.model_path)
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev(gpu=gpu))
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    
    # raise ValueError(f"Successfully loaded model and diffusion")

    logger.log("loading data...")
    # from dataset.ixi_brain import IXIBrain
    # dataset = IXIBrain(mode='test', acceleration_rate=config['dataset']['acceleration_rate'])
    # dataset = DatasetWrapper(dataset, config)
    # data = sampling_dataloader_wrapper(dataset, config)
    from dataset.ixi_brain import IXIBrain
    from dataset.fastMRI import fastMRI
    if dataset_name == "ixibrain":
        dataset = IXIBrain(mode='test', acceleration_rate=acceleration_rate, mask_pattern=mask_pattern)
        dataset = DatasetWrapper(dataset, diffusion_model_type=diffusion_model_type)
        data = sampling_dataloader_wrapper(dataset, batch_size=batch_size, num_workers=num_workers)
    elif dataset_name == "fastmri":
        # dataset = fastMRI(mode='train', acceleration_rate=acceleration_rate, mask_pattern=mask_pattern, image_size = args.image_size)
        # dataset = fastMRI(mode='test', acceleration_rate=acceleration_rate, mask_pattern=mask_pattern, image_size = args.image_size)
        dataset = fastMRI(mode='test', acceleration_rate=acceleration_rate, mask_pattern=mask_pattern, image_size = image_size, diffusion_model_type=diffusion_model_type)
        dataset = DatasetWrapper(dataset, diffusion_model_type=diffusion_model_type)
        data = sampling_dataloader_wrapper(dataset, batch_size=batch_size, num_workers=num_workers)
    else:
        raise ValueError("Check the dataset_name")
    
    if sample_conditionally == True:
        operator = get_operator(device=device, **inverse_problem_config['operator'])
        extra_measurement_params = {'sigma': inverse_problem_config['noise']['sigma']}
        measurement_noise_config = inverse_problem_config['noise']
        measurement_noise_eta = torch.tensor(measurement_noise_config['sigma'])
        combined_measurement_config = {**measurement_noise_config, **extra_measurement_params}
        noiser = get_noise(**combined_measurement_config)

        cond_method = get_conditioning_method('ps', operator, noiser)
        
        # inverse_problem_setting = measurement, measurement_cond_fn, measurement_noise_eta, x_gt


    logger.log("sampling...")
    result, low_res = [], []
    # for i in range(300):
    result = []
    low_res = []
    ground_truth = []  # 用于保存 ground truth

    for i in range(10):
        print(f"start generating the {i}th image")
        y_hat, x_gt, model_kwargs = next(data)
        print("low_res 条件输入的 shape:", model_kwargs['low_res'].shape)
        print("ground truth shape:", x_gt.shape)
        
        x_pred_for_plot = 0
        x_cond_for_plot = 0



        # 保存 ground truth（确保 tensor 在 CPU 上且转换为 numpy 数组）
        ground_truth.append(x_gt.detach().cpu().numpy())

        if sample_conditionally == True:
            if inverse_problem_config['operator']['name'] == 'fmri_reconstruction':
                measurement_cond_fn = partial(cond_method.conditioning, mask=model_kwargs['low_res'])
                inverse_problem_setting = y_hat.to(device), measurement_cond_fn, measurement_noise_eta, x_gt.to(device)
        else:
            inverse_problem_setting = None
            measurement_cond_fn = None

        if diffusion_model_type in ["measurement_conditional"]:
            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )
        elif diffusion_model_type in ["mask_conditional", "unconditional"]:
            sample_fn = (
                diffusion.langevin_stochastic_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )
        else:
            raise ValueError("Check the diffusion_model_type")

        if diffusion_model_type in ["measurement_conditional"]:
            sample = sample_fn(
                model,
                (batch_size, in_channels, image_size, image_size),
                clip_denoised=clip_denoised,
                model_kwargs={'low_res': model_kwargs['low_res'].to(device)},
                progress=True,
            )
            x_pred_for_plot = sample.clone()
            sample = sample.detach().cpu().numpy()
            result.append(sample)
            low_res.append(model_kwargs['low_res'].detach().cpu().numpy())
            x_cond_for_plot = model_kwargs['low_res'].clone()

        elif diffusion_model_type in ["mask_conditional"]:
            sample = sample_fn(
                model,
                y_hat.shape,
                mask_pattern=mask_pattern,
                acceleration_rate=acceleration_rate,
                x_clean=x_gt,
                num_iters=num_iters,
                clip_denoised=clip_denoised,
                model_kwargs={
                    'low_res': model_kwargs['low_res'].to(device),
                    'smps': model_kwargs['smps'].to(device)
                },
                progress=True,
                inverse_problem_setting=inverse_problem_setting
            )
            file_path = os.path.join(save_dir, f"{i}_sampled_image.png")
            plt.imsave(file_path, sample, cmap='gray')
            result.append(sample)
            low_res.append(model_kwargs['low_res'].detach().cpu().numpy())

        elif diffusion_model_type in ["unconditional"]:
            sample = sample_fn(
                model,
                x_gt.shape,
                mask_pattern=mask_pattern,
                acceleration_rate=acceleration_rate,
                x_clean=x_gt,
                num_iters=num_iters,
                clip_denoised=clip_denoised,
                model_kwargs=None,
                progress=True,
            )
            file_path = os.path.join(save_dir, f"{i}_sampled_image.png")
            plt.imsave(file_path, sample, cmap='gray')
            result.append(sample)

        elif diffusion_model_type in ["unconditional_del"]:
            sample = sample_fn(
                model,
                (batch_size, in_channels, image_size, image_size),
                clip_denoised=clip_denoised,
                progress=True,
            )
            file_path = os.path.join(save_dir, f"{i}_sampled_image.png")
            sample = abs_helper(sample)
            sample = sample.squeeze().detach().cpu().numpy()
            plt.imsave(file_path, sample, cmap='gray')
            result.append(sample)
        else:
            raise ValueError("Check the diffusion_model_type")

        x_start_for_plot = x_gt.clone()
        print(x_start_for_plot)

        x_start_for_plot = abs_helper(x_start_for_plot).squeeze().detach().cpu().numpy()
        x_pred_for_plot = abs_helper(x_pred_for_plot).squeeze().detach().cpu().numpy()
        x_cond_for_plot = abs_helper(x_cond_for_plot).squeeze().detach().cpu().numpy()

        # 计算PSNR和SSIM（data_range=1 表示图像数据在 [0, 1] 内）
        psnr_val = peak_signal_noise_ratio(x_start_for_plot, x_pred_for_plot, data_range=1)
        ssim_val = structural_similarity(x_start_for_plot, x_pred_for_plot, data_range=1)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(x_start_for_plot, cmap='gray')
        axes[0].set_title('Ground Truth')
        axes[0].axis('off')

        axes[1].imshow(x_pred_for_plot, cmap='gray')
        axes[1].set_title(f'Prediction\nPSNR: {psnr_val:.2f} dB\nSSIM: {ssim_val:.4f}')
        axes[1].axis('off')

        axes[2].imshow(x_cond_for_plot, cmap='gray')
        axes[2].set_title('Condition')
        axes[2].axis('off')

        plt.savefig(os.path.join("/project/cigserver4/export1/l.tingjun/output/sample_test", f"test{i}.png"))
        plt.close(fig)

    result = np.concatenate(result, 0)
    if diffusion_model_type in ["measurement_conditional", "mask_conditional"]:
        low_res = np.concatenate(low_res, 0)
        ground_truth = np.concatenate(ground_truth, 0)
    result_tensor = torch.from_numpy(result)
    result_processed = abs_helper(result_tensor).squeeze().cpu().numpy()

    if diffusion_model_type in ["measurement_conditional", "mask_conditional"]:
        low_res_tensor = torch.from_numpy(low_res)
        low_res_processed = abs_helper(low_res_tensor).squeeze().cpu().numpy()
        
        ground_truth_tensor = torch.from_numpy(ground_truth)
        ground_truth_processed = abs_helper(ground_truth_tensor).squeeze().cpu().numpy()
        
        tifffile.imwrite(os.path.join(save_dir, 'sampled_img.tiff'), result_processed)
        tifffile.imwrite(os.path.join(save_dir, 'low_res.tiff'), low_res_processed)
        tifffile.imwrite(os.path.join(save_dir, 'real_ground_truth.tiff'), ground_truth_processed)
    else:
        tifffile.imwrite(os.path.join(save_dir, 'sampled_img.tiff'), result_processed)

    if diffusion_model_type in ["measurement_conditional", "mask_conditional"]:
        low_res = np.concatenate(low_res, 0)
        ground_truth = np.concatenate(ground_truth, 0)

    # final_x_gt = ground_truth[0]

    #   final_x_gt_tensor = torch.tensor(final_x_gt).view(1, 2, 256, 256)

    # print(final_x_gt_tensor.shape)

    # fig, axes = plt.subplots(1, 1, figsize=(15, 5))
    # final_x_gt_tensor = abs_helper(final_x_gt_tensor).squeeze().numpy()
    # axes.imshow(final_x_gt_tensor, cmap='gray')
    # axes.set_title('Ground Truth')
    # axes.axis('off')
    # plt.savefig(os.path.join("/project/cigserver4/export1/l.tingjun/output/sample_test", f"TTT.png"))
    # plt.close(fig)

    # tifffile.imwrite(os.path.join(save_dir, 'sampled_img.tiff'), result)
    # tifffile.imwrite(os.path.join(save_dir, 'low_res.tiff'), low_res)
    # tifffile.imwrite(os.path.join(save_dir, 'real_ground_truth.tiff'), ground_truth)

