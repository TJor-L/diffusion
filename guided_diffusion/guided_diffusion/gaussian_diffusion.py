"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math

import numpy as np
import torch as th
import torch

from .nn import mean_flat, compute_mse_loss
from .losses import normal_kl, discretized_gaussian_log_likelihood
import matplotlib.pyplot as plt
from utility.tweedie_utility import clear_color, get_noiselevel_alphas_timestep, extract_and_expand_value, compute_metrics
from utility.utility import crop_images, abs_helper
import os

from dataset.fastMRI import fmult, ftran, ftran_non_mask, fmult_non_mask, randomly_cartesian_mask, uniformly_cartesian_mask, mix_cartesian_mask, np_torch_renormalize

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):
        self.showed_image = 0
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, mask=None, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        # print("in pmean")
        # print(model_kwargs)
        # print(type(model))
        assert t.shape == (B,)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)
        # print("iaft")
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            if mask != None:
                degraded_pred_xstart = apply_mask_on_kspace(pred_xstart, mask)
                model_mean, _, _ = self.q_posterior_mean_variance(
                    x_start=degraded_pred_xstart, x_t=x, t=t
                )
            else:
                model_mean, _, _ = self.q_posterior_mean_variance(
                    x_start=pred_xstart, x_t=x, t=t
                )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, self._scale_timesteps(t), **model_kwargs
        )

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        # print("YEEEEEP")
        # print(cond_fn)
        # print( model_kwargs['low_res'].shape)
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            print("YEEEEEP")
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def langevin_stochastic_sample_loop(
        self,
        model,
        shape,
        mask_pattern,
        acceleration_rate,
        num_iters = 1000,
        x_clean = None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        inverse_problem_setting = None
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        all_sample = []
        for sample in self.langevin_stochastic_sample_loop_progressive(
            model,
            shape,
            mask_pattern = mask_pattern,
            acceleration_rate = acceleration_rate,
            num_iters = num_iters,
            inverse_problem_setting = inverse_problem_setting,
            x_clean = x_clean,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            # all_sample.append(sample['sample'].detach().cpu().squeeze().numpy())
            # all_sample.append(sample.detach().cpu().squeeze().numpy())
            final = sample
            # print(f"len(all.sample): {len(all_sample)}")
        # all_sample = np.stack(all_sample, 0)
        return final

    def langevin_stochastic_sample_loop_progressive(
        self,
        model,
        shape,
        mask_pattern,
        acceleration_rate,
        num_iters = 1000,
        inverse_problem_setting = None,
        x_clean = None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        from tqdm import tqdm
        
        compute_likelihood_in_dps = True

        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))

        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)#.requires_grad_()
        
        import torch
        
        if len(img.shape) == 5:
            is_state_variable_measurement = True
            smps = model_kwargs["smps"]
            smps = smps.permute([0, 1, 4, 2, 3]).contiguous().squeeze(0).squeeze(0)
            measurement_mask = model_kwargs["low_res"]
            measurement_mask = measurement_mask.squeeze(0)
        else:
            is_state_variable_measurement = False
        
        if inverse_problem_setting != None:
            y_hat, measurement_cond_fn, measurement_noise_eta, x_gt = inverse_problem_setting
            
            y_hat_imgspace = from_kspace_to_image(y_hat, smps)
            input_psnr_value, input_ssim_value = compute_metrics(reconstructed = y_hat_imgspace, reference = x_gt)
            
        assert self.num_timesteps == 1000
        extended_denoiser_noise_sigma_array, extended_alphas_array, time_array, time_idx_array =  get_noiselevel_alphas_timestep(beta_at_clean = 0.0001, schedule_name = "denoiser", num_diffusion_timesteps = self.num_timesteps, last_time_step = 0, num_iters = num_iters, save_plot=True)
        
        reverse_time_array = time_array[::-1]
        reverse_time_idx_array = time_idx_array[::-1]
        reverse_alphas_array = extended_alphas_array[::-1]
        reverse_noise_sigma_array = extended_denoiser_noise_sigma_array[::-1]

        assert len(time_array) == len(time_idx_array) == len(extended_denoiser_noise_sigma_array) == len(extended_alphas_array)
               
        time_alpha_noisesigma_step_list = list(zip(reverse_time_idx_array, reverse_time_array, reverse_alphas_array, reverse_noise_sigma_array))
        pbar = tqdm(time_alpha_noisesigma_step_list)
        for loop_index, (index_reverse_time_str, indexed_reverse_time_str, indexed_reverse_alphas, indexed_reverse_noise_sigma) in enumerate(pbar):
            if is_state_variable_measurement == True:
                # raise ValueError(f"mask_pattern: {mask_pattern}")
                if mask_pattern == "randomly_cartesian":
                    new_mask = (torch.tensor((randomly_cartesian_mask((shape[2], shape[3]), acceleration_rate))).unsqueeze(0).unsqueeze(0)).to(device)
                elif mask_pattern == "uniformly_cartesian":
                    new_mask = (torch.tensor((uniformly_cartesian_mask((shape[2], shape[3]), acceleration_rate))).unsqueeze(0).unsqueeze(0)).to(device)
                elif mask_pattern == "mix_cartesian":
                    new_mask = (torch.tensor((mix_cartesian_mask((shape[2], shape[3]), acceleration_rate))).unsqueeze(0).unsqueeze(0)).to(device)
                else:
                    raise ValueError("mask_pattern should be either 'randomly_cartesian' or 'uniformly_cartesian', 'mix_cartesian'")
                model_kwargs['low_res'] = new_mask
                
            t = indexed_reverse_time_str
            time = torch.tensor([t] * img.shape[0], device=device)#.clone().detach() # * e.g) time: tensor([999]) at the first index
            
            # print(f"t: {t}")
            alphas_coef = extract_and_expand_value(indexed_reverse_alphas, t, img).to(device)
            
            img = img.requires_grad_()
            # TODO: HERE I can reinitialize the mask entry of model_kwargs
            
            model_output = model(torch.sqrt(alphas_coef) * img, time, **model_kwargs) if is_state_variable_measurement == True else model(torch.sqrt(alphas_coef) * img, time, model_kwargs)
            if model_output.shape[1] == 2 * x_clean.shape[1]:
                model_output, model_var_values = torch.split(model_output, x_clean.shape[1], dim=1)
                
            if is_state_variable_measurement == True:
                model_output = model_output.permute([0, 2, 3, 1]).contiguous().squeeze(0)
                model_output = torch.view_as_complex(model_output)
                model_output = fmult_non_mask(model_output, smps)
                model_output = torch.view_as_real(model_output)
                model_output = model_output.permute([3, 1, 2, 0]).contiguous().unsqueeze(0)
            
            if inverse_problem_setting != None:
                # pred_xstart = self._predict_xstart_from_eps(img, time, model_output)
                pred_xstart = torch.sqrt(1/(alphas_coef))*(img) - torch.sqrt(1/alphas_coef - 1)*model_output
                # raise ValueError(f"pred_xstart: {pred_xstart}")
            else:
                pred_xstart = torch.sqrt(1/(alphas_coef))*(img) - torch.sqrt(1/alphas_coef - 1)*model_output
            
            alphas_coef_for_model_output = extract_and_expand_value(indexed_reverse_alphas, t, model_output).to(device)
            # score = - torch.sqrt((1)/(1-alphas_coef_for_model_output)) * model_output * torch.sqrt(alphas_coef_for_model_output)
            score = - torch.sqrt((1)/(1-alphas_coef)) * model_output * torch.sqrt(alphas_coef)
            noise_sigma_square = torch.square(torch.tensor(indexed_reverse_noise_sigma))
            
            # print(f"t: {t}")
            # print(f"noise_sigma_square: {noise_sigma_square}")

            noise_k_square = torch.square(torch.tensor(reverse_noise_sigma_array[loop_index]))
            noise_k_next_square = torch.square(torch.tensor(reverse_noise_sigma_array[loop_index+1])) if loop_index < num_iters-1 else noise_k_square
            # print(f"noise_k_square: {noise_k_square}")
            # print(f"noise_k_next_square: {noise_k_next_square}")
            assert noise_k_square >= noise_k_next_square

            lgv_score_x_coefficient = 1
            
            langevin_step_size = noise_k_square - noise_k_next_square
            # if langevin_step_size > torch.square(torch.tensor(0.05)):
            #     langevin_step_size = torch.square(torch.tensor(0.05))
            lgv_score_x_hat_coefficient = langevin_step_size
            lgv_score_noise_coefficient = torch.sqrt(langevin_step_size)
            noise_N = torch.randn_like(model_output)

            # img = img.permute([0, 4, 2, 3, 1]).contiguous()
            # img = th.view_as_complex(img)
            # print(f"unet.py x.shape: {x.shape} \n smps.shape: {smps.shape} \n low_res.shape: {low_res.shape}")
            # img = ftran_non_mask(img, model_kwargs['smps'].permute([0, 1, 4, 2, 3]).contiguous().squeeze(0).squeeze(0))
            # img = th.view_as_real(img).permute([0, 3, 1, 2]).contiguous()
            if (loop_index != num_iters - 1):
                # raise ValueError(f"img.shape: {img.shape}\nscore.shape: {score.shape}\nnoise_N.shape: {noise_N.shape}")
                # print(f"A")
                img_score = lgv_score_x_coefficient * img + lgv_score_x_hat_coefficient * score + lgv_score_noise_coefficient * noise_N                        
            else:
                # print(f"B")
                img_score = lgv_score_x_coefficient * img + lgv_score_x_hat_coefficient * score
                
            # ------------
            # Doing conditional Langevin update if solving inverse problems
            # ------------
            if inverse_problem_setting != None:
                norm_grad, distance, _ = measurement_cond_fn(x_t= img,
                                    measurement=y_hat,
                                    noisy_measurement=y_hat,
                                    x_prev=img,
                                    x_0_hat=pred_xstart,
                                    # x_0_hat=img,
                                    smps = smps,
                                    mask = measurement_mask)
                # print(f"norm_grad: {norm_grad}")
                if compute_likelihood_in_dps == True:
                    measurement_noise_eta_square = torch.square(measurement_noise_eta)
                    lgv_likelihood_coefficient = -1. * 0.6
                    img_cond = img_score + lgv_likelihood_coefficient * norm_grad
                    img = img_cond
                else:
                    measurement_noise_eta_square = torch.square(measurement_noise_eta)
                    lgv_likelihood_coefficient = -1. * langevin_step_size * (1/(measurement_noise_eta_square))
                    img_cond = img_score + lgv_likelihood_coefficient * norm_grad
                    img = img_cond
            else:
                img = img_score

            img = img.detach()
            img_score = img_score.detach()
                
            if (loop_index % 10 == 0) or loop_index == num_iters - 1: 
                # raise ValueError(f"HERE")
                if is_state_variable_measurement == True:
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    img_for_plot = img.clone()
                    img_for_plot = img_for_plot.permute([0, 4, 2, 3, 1]).contiguous()
                    img_for_plot = th.view_as_complex(img_for_plot)
                    # print(f"unet.py x.shape: {x.shape} \n smps.shape: {smps.shape} \n low_res.shape: {low_res.shape}")
                    img_for_plot = ftran_non_mask(img_for_plot, smps)
                    img_for_plot = th.view_as_real(img_for_plot).permute([0, 3, 1, 2]).contiguous()
                    img_for_plot = img_for_plot.detach()
                    # img_for_plot = np_torch_renormalize(img_for_plot)

                    pred_xstart_for_plot = pred_xstart.clone()
                    pred_xstart_for_plot = pred_xstart_for_plot.detach()
                    pred_xstart_for_plot = from_kspace_to_image(pred_xstart_for_plot, smps)
                    pred_xstart_for_plot = abs_helper(pred_xstart_for_plot).squeeze().detach().cpu().numpy()
                    

                    mask_for_plot = new_mask.clone()
                    mask_for_plot = mask_for_plot.detach().squeeze().detach().cpu().numpy()

                    img_for_plot = abs_helper(img_for_plot).squeeze().detach().cpu().numpy()
                    axes[0].imshow(img_for_plot, cmap='gray')
                    axes[0].set_title(f'x_(k={loop_index})')
                    axes[0].axis('off')
                    
                    axes[1].imshow(mask_for_plot, cmap='gray')
                    axes[1].set_title(f'Mask_(k={loop_index})')
                    axes[1].axis('off')

                    axes[2].imshow(pred_xstart_for_plot, cmap='gray')
                    axes[2].set_title(f'Pred x start(k={loop_index})')
                    axes[2].axis('off')
                    
                    plt.savefig(os.path.join(".", "sampling_progress.png"))
                    plt.close(fig)
                else:
                    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
                    img_for_plot = img.clone()
                    img_for_plot = img_for_plot.detach()
                    img_for_plot = abs_helper(img_for_plot).squeeze().detach().cpu().numpy()
                    axes.imshow(img_for_plot, cmap='gray')
                    axes.set_title(f'x_(k={loop_index})')
                    axes.axis('off')
                    
                    plt.savefig(os.path.join(".", "sampling_progress.png"))
                    plt.close(fig)
                
            # ------------
            # Put img back to the measurement domain
            # ------------
            # if loop_index == num_iters - 1:
            #     pass
            # else:
            #     img = (th.view_as_complex(img.permute([0, 2, 3, 1]).contiguous())).squeeze(0)
            #     img = fmult_non_mask(img, model_kwargs['smps'].permute([0, 1, 4, 2, 3]).contiguous().squeeze(0).squeeze(0))
            #     img = (th.view_as_real(img).permute([3, 1, 2, 0]).contiguous()).unsqueeze(0)
            # # ! Compute PSNR
            yield img_for_plot
            # yield img

            # img = img.detach()
        # return img_for_plot
            
    def p_sample_loop(
        self,
        model,
        shape,
        x_clean = None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        all_sample = []
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            x_clean = x_clean,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            all_sample.append(sample['sample'].detach().cpu().squeeze().numpy())
            final = sample
        all_sample = np.stack(all_sample, 0)
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        x_clean = None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        print("in loop pro")
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
            # raise ValueError(f"device: {device}")
            # print("HERE WE ARE")
            # img = self.q_sample(x_clean.to(device), th.tensor(999).to(device))
            
            
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]
                
                fig, axes = plt.subplots(1, 1, figsize=(5, 5))
                img_for_plot = img.clone()
                img_for_plot = img_for_plot.detach()

                img_for_plot = abs_helper(img_for_plot).squeeze().detach().cpu().numpy()
                axes.imshow(img_for_plot, cmap='gray')
                axes.set_title(f'x_(k={i})')
                axes.axis('off')
                
                plt.savefig(os.path.join(".", "sampling_progress.png"))
                plt.close(fig)

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    def _vb_terms_bpd(
        self, model, x_start, x_t, t, mask = None, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        # raise ValueError(f"x_start.shape: {x_start.shape}\nx_t.shape: {x_t.shape}\nt.shape: {t.shape}")
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, mask = mask, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        
        # if mask != None:
        #     # print(f"out['mean'].shape: {out['mean'].shape}\nmask.shape: {mask.shape}\nout['log_variance'].shape: {out['log_variance'].shape}")
        #     """
        #     out.shape: [1, 2, 320, 320, 20]
        #     """
        #     out['mean'] = apply_mask_on_kspace(out['mean'], mask)
        #     print(f"mask has been applied")
        
        # TODO: In my opinion, I can apply the degradation operator to the out["mean"]
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, x_start, t, acceleration_rate, compute_vb_loss, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}
        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            # print("model_kwargs:", model_kwargs)
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)
          
            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert (model_output.shape == (B, C * 2, *x_t.shape[2:])) or (x_t.shape[-1] == 20 and model_output.shape[1] == x_t.shape[1]*2)
                
                model_output, model_var_values = th.split(model_output, C, dim=1)

                # print(f"x_t.shape: {x_t.shape}\nmodel_output.shape: {model_output.shape}")
                pred_xstart = self._predict_xstart_from_eps(x_t, t, model_output)
                # raise ValueError(f"pred_xstart.shape: {pred_xstart.shape}")
                
                x_start_for_plot = x_start.clone()
                x_t_for_plot = x_t.clone()
                x_low_res_for_plot = model_kwargs['low_res'].clone()
                    
                # pred_xstart_for_plot = self._predict_xstart_from_eps(x_t_for_plot, t, model_output)
                pred_xstart_for_plot = pred_xstart.clone()
                pred_xstart_for_plot = pred_xstart_for_plot.detach()
                # --------
                # Plot the figure of x_t, x_start, pred_xstart
                # --------
                # if t % 10 == 0:
                if ((t < 100) and (t % 2 == 0) and (t != 0)) or ((t >= 100) and (t % 5 == 0)):
                    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                    x_start_for_plot = abs_helper(x_start_for_plot).squeeze().detach().cpu().numpy()
                    axes[0].imshow(x_start_for_plot, cmap='gray')
                    axes[0].set_title('Ground Truth')
                    axes[0].axis('off')
                    
                    x_t_for_plot = abs_helper(x_t_for_plot).squeeze().detach().cpu().numpy()
                    axes[1].imshow(x_t_for_plot, cmap='gray')
                    axes[1].set_title(f'Noisy ({t.item()})')
                    axes[1].axis('off')
                    
                    pred_xstart_for_plot = abs_helper(pred_xstart_for_plot).squeeze().detach().cpu().numpy()
                    axes[2].imshow(pred_xstart_for_plot, cmap='gray')
                    axes[2].set_title('Prediction x start')
                    axes[2].axis('off')

                    x_low_res_for_plot = abs_helper(x_low_res_for_plot).squeeze().detach().cpu().numpy()
                    axes[3].imshow(x_low_res_for_plot, cmap='gray')
                    axes[3].set_title('Condition x_low_res')
                    axes[3].axis('off')

                    plt.savefig(os.path.join("/project/cigserver4/export1/l.tingjun/training_img/cond1", f"imgspace_fig_training_{acceleration_rate}_{t.item()}_{self.showed_image}.png"))
                    self.showed_image+=1
                    plt.close(fig)
                
                
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                
                # TODO: Modify the KL divergence considerations
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            
            # ---------------
            # Apply degradation operator to the x_start and model_output
            # ---------------
            """
            if condition_on_mask == True:
                target = target.permute([0, 4, 2, 3, 1]).contiguous()
                target = th.view_as_complex(target)
                target = target * model_kwargs["low_res"]
                target = th.view_as_real(target)
                target = target.permute([0, 4, 2, 3, 1]).contiguous()
            """
            # raise ValueError(f"target.shape: {target.shape}")
            # 1, 2, 320, 320, 20
            # print(f"target.shape: {target.shape}\nmodel_output.shape: {model_output.shape}")

            # TODO: using model_output to obtain the prediction of x_start
            
            # print(f"x_start.shape: {x_start.shape}\nmodel_output.shape: {model_output.shape}\nnoise:{noise.shape}")
            
            # new_xstart = self._predict_xstart_from_eps(x_t, t, noise)
            
            # terms["mse"] = mean_flat((new_xstart - pred_xstart) ** 2)
            # terms["mse"] = mean_flat((x_start - pred_xstart) ** 2)
            # print(f"_extract_into_tensor(self.alphas_cumprod, t, x_t.shape): {_extract_into_tensor(self.alphas_cumprod, torch.tensor(999).to('cuda:3'), x_t.shape)}")
            # print(f"_extract_into_tensor(self.alphas_cumprod, t, x_t.shape): {_extract_into_tensor(self.alphas_cumprod, torch.tensor(1).to('cuda:3'), x_t.shape)}")
            alphas_coef = _extract_into_tensor(self.alphas_cumprod, t, x_t.shape)
            
            targetMMSE = (x_t - (torch.sqrt(1-alphas_coef))*target)/((torch.sqrt(alphas_coef)))
            predictedMMSE = (x_t - (torch.sqrt(1-alphas_coef))*model_output)/((torch.sqrt(alphas_coef)))
            
            weightconstant_sigma = torch.sqrt((1-alphas_coef) / alphas_coef)
            weightconstant = 1/torch.square(weightconstant_sigma)
            
            # first_term = (x_t - (torch.sqrt(1-alphas_coef))*target)/(torch.sqrt(alphas_coef))
            # second_term = (x_t - (torch.sqrt(1-alphas_coef))*model_output)/(torch.sqrt(alphas_coef))
            # first_term = (x_t - (torch.sqrt(1-alphas_coef))*target)/((torch.sqrt(alphas_coef)))
            # second_term = (x_t - (torch.sqrt(1-alphas_coef))*model_output)/((torch.sqrt(alphas_coef)))
            
            # print(f"torch.max(first_term): {torch.max(first_term)}\ntorch.mean(first_term): {torch.mean(first_term)}\ntorch.min(first_term): {torch.min(first_term)}")
            # print(f"torch.max(second_term): {torch.max(second_term)}\ntorch.mean(second_term): {torch.mean(second_term)}\ntorch.min(second_term): {torch.min(second_term)}")
            
            # print(f"torch.max(target): {torch.max(target)}\ntorch.mean(target): {torch.mean(target)}\ntorch.min(target): {torch.min(target)}")
            # print(f"torch.max(model_output): {torch.max(model_output)}\ntorch.mean(model_output): {torch.mean(model_output)}\ntorch.min(model_output): {torch.min(model_output)}")
            
            terms["mse"] = mean_flat(weightconstant*((targetMMSE - predictedMMSE) ** 2))
            # terms["mse"] = mean_flat((target - model_output) ** 2)
            # terms["mse"] = mean_flat((first_term - second_term) ** 2)
            # terms["mse"] = compute_mse_loss(predicted = second_term, target = first_term)

            if "vb" in terms:
                if compute_vb_loss == True:
                    terms["loss"] = terms["mse"] + terms["vb"]
                else:
                    terms["loss"] = terms["mse"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms
    
    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def from_kspace_to_image(kspace, smps):
    kspace = kspace.permute([0, 4, 2, 3, 1]).contiguous()
    kspace = th.view_as_complex(kspace)
    kspace = ftran_non_mask(kspace, smps)
    kspace = th.view_as_real(kspace).permute([0, 3, 1, 2]).contiguous()
    return kspace

def from_image_to_kspace(image, smps):
    image = image.permute([0, 2, 3, 1]).contiguous().squeeze(0)
    image = torch.view_as_complex(image)
    image = fmult_non_mask(image, smps)
    image = image.squeeze(0)
    # TODO: Multipling the mask to the model_output
    image = torch.view_as_real(image)
    image = image.permute([3, 1, 2, 0]).contiguous().unsqueeze(0)
    return image

def apply_mask_on_kspace(kspace, mask):
    kspace = kspace.permute([0, 4, 2, 3, 1]).contiguous()
    kspace = th.view_as_complex(kspace)
    kspace = kspace * mask
    kspace = th.view_as_real(kspace).permute([0, 4, 2, 3, 1]).contiguous()
    return kspace