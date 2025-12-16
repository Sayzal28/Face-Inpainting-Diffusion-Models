"""
Gaussian diffusion process for training and sampling with advanced inpainting injection.
"""

import torch
import numpy as np
from tqdm.auto import tqdm

from losses import ModelMeanType, ModelVarType, LossType, normal_kl, discretized_gaussian_log_likelihood, mean_flat


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of tensors to extract from.
    :return: a tensor with the same shape as broadcast_shape.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models with advanced inpainting injection.

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                             model so that they are always scaled like in the
                             original paper (0 to 1000).
    """
    
    def __init__(self, *, betas, model_mean_type, model_var_type, loss_type, rescale_timesteps=False):
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
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )

        # Cache for ground truth noises (for advanced inpainting)
        self._gt_noises_cache = {}

    def get_gt_noised(self, gt, timestep):
        """
        Get properly noised ground truth for a specific timestep.
        Uses caching to ensure consistent noise across sampling steps.
        
        :param gt: Ground truth image [B, C, H, W]
        :param timestep: Current timestep (int)
        :return: Noised ground truth at the specified timestep
        """
        cache_key = (gt.shape, timestep, gt.device)
        
        if cache_key not in self._gt_noises_cache:
            # Generate noise once and cache it
            noise = torch.randn_like(gt)
            self._gt_noises_cache[cache_key] = noise
        else:
            noise = self._gt_noises_cache[cache_key]
        
        # Create timestep tensor
        t = torch.tensor([timestep], device=gt.device).expand(gt.shape[0])
        
        # Add noise according to diffusion schedule
        noised_gt = self.q_sample(gt, t, noise=noise)
        return noised_gt

    def clear_gt_noise_cache(self):
        """Clear the ground truth noise cache (call between different images)."""
        self._gt_noises_cache.clear()

    def apply_inpainting_injection(self, x, t, gt, gt_keep_mask, 
                                  use_cumulative_noise=True, injection_schedule="all"):
        """
        Apply advanced inpainting injection during sampling.
        
        :param x: Current sample [B, C, H, W]
        :param t: Current timestep tensor [B]
        :param gt: Ground truth image [B, C, H, W]
        :param gt_keep_mask: Mask for known regions [B, 1, H, W] (1 = keep, 0 = generate)
        :param use_cumulative_noise: If True, use cumulative noise schedule
        :param injection_schedule: When to apply injection ("all", "high", "low")
        :return: x with injected ground truth in known regions
        """
        if gt is None or gt_keep_mask is None:
            return x
        
        # Check injection schedule
        timestep = int(t[0].item())
        if injection_schedule == "high" and timestep < self.num_timesteps // 2:
            return x  # Only inject in high noise timesteps
        elif injection_schedule == "low" and timestep >= self.num_timesteps // 2:
            return x  # Only inject in low noise timesteps
        
        if use_cumulative_noise:
            # Use cached noise for consistency
            weighed_gt = self.get_gt_noised(gt, timestep)
        else:
            # Use current noise level
            alpha_cumprod = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
            gt_weight = torch.sqrt(alpha_cumprod)
            noise_weight = torch.sqrt(1 - alpha_cumprod)
            
            # Generate fresh noise
            noise = torch.randn_like(gt)
            weighed_gt = gt_weight * gt + noise_weight * noise
        
        # Expand mask to all channels if needed
        if gt_keep_mask.shape[1] == 1 and x.shape[1] > 1:
            gt_keep_mask = gt_keep_mask.repeat(1, x.shape[1], 1, 1)
        
        # Inject: keep known regions from noised GT, generated regions from x
        x_injected = gt_keep_mask * weighed_gt + (1 - gt_keep_mask) * x
        
        return x_injected

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
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
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
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
            posterior_mean.shape[0] == posterior_variance.shape[0] == 
            posterior_log_variance_clipped.shape[0] == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
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
        assert t.shape == (B,)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = torch.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = torch.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = torch.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
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
            model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
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
        return (
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            ) * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart
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

    def p_sample(self, model, x, t, clip_denoised=True, denoised_fn=None, cond_fn=None, 
                 model_kwargs=None, use_inpainting_injection=False, 
                 injection_schedule="all", use_cumulative_noise=True):
        """
        Sample x_{t-1} from the model at the given timestep with optional inpainting injection.
        
        :param use_inpainting_injection: Whether to apply inpainting injection
        :param injection_schedule: When to apply injection ("all", "high", "low")
        :param use_cumulative_noise: Whether to use cumulative noise schedule
        """
        # Apply inpainting injection before model prediction if enabled
        if use_inpainting_injection and model_kwargs:
            gt = model_kwargs.get('gt')
            gt_keep_mask = model_kwargs.get('gt_keep_mask')
            if gt is not None and gt_keep_mask is not None:
                x = self.apply_inpainting_injection(
                    x, t, gt, gt_keep_mask, 
                    use_cumulative_noise=use_cumulative_noise,
                    injection_schedule=injection_schedule
                )
        
        out = self.p_mean_variance(
            model, x, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn, model_kwargs=model_kwargs,
        )
        noise = torch.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(cond_fn, out, x, t, model_kwargs=model_kwargs)
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(self, model, shape, noise=None, clip_denoised=True, denoised_fn=None,
                      cond_fn=None, model_kwargs=None, device=None, progress=False,
                      use_inpainting_injection=False, injection_schedule="all", 
                      use_cumulative_noise=True):
        """
        Generate samples from the model with optional advanced inpainting injection.
        
        :param use_inpainting_injection: Whether to apply inpainting injection
        :param injection_schedule: When to apply injection ("all", "high", "low")
        :param use_cumulative_noise: Whether to use cumulative noise schedule
        """
        # Clear noise cache at start of sampling
        self.clear_gt_noise_cache()
        
        final = None
        for sample in self.p_sample_loop_progressive(
            model, shape, noise=noise, clip_denoised=clip_denoised, denoised_fn=denoised_fn,
            cond_fn=cond_fn, model_kwargs=model_kwargs, device=device, progress=progress,
            use_inpainting_injection=use_inpainting_injection,
            injection_schedule=injection_schedule,
            use_cumulative_noise=use_cumulative_noise
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(self, model, shape, noise=None, clip_denoised=True,
                                  denoised_fn=None, cond_fn=None, model_kwargs=None, device=None, 
                                  progress=False, use_inpainting_injection=False,
                                  injection_schedule="all", use_cumulative_noise=True):
        """
        Generate samples from the model and yield intermediate samples from each timestep of diffusion.
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            indices = tqdm(indices)

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self.p_sample(
                    model, img, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn,
                    cond_fn=cond_fn, model_kwargs=model_kwargs,
                    use_inpainting_injection=use_inpainting_injection,
                    injection_schedule=injection_schedule,
                    use_cumulative_noise=use_cumulative_noise
                )
                yield out
                img = out["sample"]

    def ddim_sample(self, model, x, t, clip_denoised=True, denoised_fn=None, cond_fn=None,
                    model_kwargs=None, eta=0.0, use_inpainting_injection=False,
                    injection_schedule="all", use_cumulative_noise=True):
        """
        Sample x_{t-1} from the model using DDIM with optional inpainting injection.
        """
        # Apply inpainting injection before model prediction if enabled
        if use_inpainting_injection and model_kwargs:
            gt = model_kwargs.get('gt')
            gt_keep_mask = model_kwargs.get('gt_keep_mask')
            if gt is not None and gt_keep_mask is not None:
                x = self.apply_inpainting_injection(
                    x, t, gt, gt_keep_mask,
                    use_cumulative_noise=use_cumulative_noise,
                    injection_schedule=injection_schedule
                )
        
        out = self.p_mean_variance(
            model, x, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn, model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * 
            torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        noise = torch.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * torch.sqrt(alpha_bar_prev) + 
            torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(self, model, shape, noise=None, clip_denoised=True, denoised_fn=None,
                         cond_fn=None, model_kwargs=None, device=None, progress=False, eta=0.0,
                         use_inpainting_injection=False, injection_schedule="all",
                         use_cumulative_noise=True):
        """
        Generate samples from the model using DDIM with optional advanced inpainting injection.
        """
        # Clear noise cache at start of sampling
        self.clear_gt_noise_cache()
        
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model, shape, noise=noise, clip_denoised=clip_denoised, denoised_fn=denoised_fn,
            cond_fn=cond_fn, model_kwargs=model_kwargs, device=device, progress=progress, eta=eta,
            use_inpainting_injection=use_inpainting_injection,
            injection_schedule=injection_schedule,
            use_cumulative_noise=use_cumulative_noise
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(self, model, shape, noise=None, clip_denoised=True,
                                     denoised_fn=None, cond_fn=None, model_kwargs=None, device=None,
                                     progress=False, eta=0.0, use_inpainting_injection=False,
                                     injection_schedule="all", use_cumulative_noise=True):
        """
        Use DDIM to sample from the model and yield intermediate samples from each timestep of DDIM.
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            indices = tqdm(indices)

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self.ddim_sample(
                    model, img, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn,
                    cond_fn=cond_fn, model_kwargs=model_kwargs, eta=eta,
                    use_inpainting_injection=use_inpainting_injection,
                    injection_schedule=injection_schedule,
                    use_cumulative_noise=use_cumulative_noise
                )
                yield out
                img = out["sample"]

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None,
                    use_injection=True, injection_schedule="all", use_cumulative_noise=True):
        """
        Compute training losses with optional explicit inpainting injection.

        Args:
            model: The model to evaluate.
            x_start: Clean images [B, C, H, W]
            t: Timesteps [B]
            model_kwargs: Should contain 'mask' and 'masked_image'
            noise: Optional noise to use
            use_injection: If True, apply inpainting injection like in sampling
            injection_schedule: Injection mode ("all", "high", "low")
            use_cumulative_noise: Whether to use cumulative noise schedule
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)

        mask = model_kwargs.get("mask")
        masked_image = model_kwargs.get("masked_image")

        if mask is None:
            mask = torch.ones(x_start.shape[0], 1, x_start.shape[2], x_start.shape[3], 
                          device=x_start.device)

        # Forward diffusion - add noise
        x_t = self.q_sample(x_start, t, noise=noise)

        # -------------------------------------------------
        # 🔹 NEW: Explicit injection of known regions
        if use_injection and masked_image is not None:
            # gt_keep_mask: 1 = keep, 0 = inpaint
            gt_keep_mask = 1 - mask
            x_t = self.apply_inpainting_injection(
                x=x_t,
            t=t,
            gt=x_start,
            gt_keep_mask=gt_keep_mask,
            use_cumulative_noise=use_cumulative_noise,
            injection_schedule=injection_schedule
        )
    

        terms = {}

        if self.loss_type in [LossType.MSE, LossType.RESCALED_MSE]:
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = torch.split(model_output, C, dim=1)

            target = noise
            mask_3ch = mask.repeat(1, 3, 1, 1)

            mse_loss = (target - model_output) ** 2
            masked_mse = mse_loss * mask_3ch

            mask_area = mask_3ch.sum(dim=[2, 3], keepdim=True)
            mask_area = torch.clamp(mask_area, min=1.0)

            terms["mse"] = masked_mse.sum(dim=[2, 3], keepdim=True) / mask_area
            terms["mse"] = terms["mse"].mean()

            if self.loss_type == LossType.RESCALED_MSE:
                terms["mse"] *= self.num_timesteps

            terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(f"Loss type {self.loss_type} not implemented for masking")

        return terms


    def _vb_terms_bpd(self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None):
        """
        Get a term for the variational lower-bound.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(true_mean, true_log_variance_clipped, out["mean"], out["log_variance"])
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        output = torch.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    # Convenience methods for easy usage
    def sample_with_advanced_inpainting(self, model, shape, gt=None, gt_keep_mask=None,
                                       use_ddim=True, eta=0.0, progress=True, device=None,
                                       injection_schedule="all", use_cumulative_noise=True):
        """
        Convenience method for sampling with advanced inpainting injection.
        
        :param model: The diffusion model
        :param shape: Shape of samples to generate
        :param gt: Ground truth image [B, C, H, W]
        :param gt_keep_mask: Keep mask [B, 1, H, W] (1 = keep, 0 = generate)
        :param use_ddim: Whether to use DDIM sampling
        :param eta: DDIM eta parameter
        :param progress: Show progress bar
        :param device: Device to sample on
        :param injection_schedule: When to inject ("all", "high", "low")
        :param use_cumulative_noise: Whether to use cumulative noise schedule
        :return: Generated samples
        """
        if device is None:
            device = next(model.parameters()).device
        
        # Setup model kwargs for inpainting
        model_kwargs = {}
        use_injection = False
        
        if gt is not None and gt_keep_mask is not None:
            model_kwargs['gt'] = gt
            model_kwargs['gt_keep_mask'] = gt_keep_mask
            use_injection = True
            
            # Also add standard inpainting inputs
            # NEW CONVENTION: gt_keep_mask has 1=keep, 0=generate
            # So inpaint_mask should be 1-gt_keep_mask (1=inpaint, 0=keep)
            masked_image = gt * gt_keep_mask  # Keep only the white regions (1s)
            inpaint_mask = 1 - gt_keep_mask   # Convert to inpaint mask (1=inpaint, 0=keep)
            model_kwargs['masked_image'] = masked_image
            model_kwargs['mask'] = inpaint_mask
        
        if use_ddim:
            return self.ddim_sample_loop(
                model=model,
                shape=shape,
                device=device,
                progress=progress,
                eta=eta,
                model_kwargs=model_kwargs,
                use_inpainting_injection=use_injection,
                injection_schedule=injection_schedule,
                use_cumulative_noise=use_cumulative_noise
            )
        else:
            return self.p_sample_loop(
                model=model,
                shape=shape,
                device=device,
                progress=progress,
                model_kwargs=model_kwargs,
                use_inpainting_injection=use_injection,
                injection_schedule=injection_schedule,
                use_cumulative_noise=use_cumulative_noise
            )
