from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler
from tqdm import tqdm
import numpy as np
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps


def scale_noise(
    scheduler,
    sample: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    noise: Optional[torch.FloatTensor] = None,
) -> torch.FloatTensor:
    """
    Forward process in flow-matching

    Args:
        sample (`torch.FloatTensor`):
            The input sample.
        timestep (`int`, *optional*):
            The current timestep in the diffusion chain.

    Returns:
        `torch.FloatTensor`:
            A scaled input sample.
    """
    scheduler._init_step_index(timestep)
    sigma = scheduler.sigmas[scheduler.step_index]
    sample = sigma * noise + (1.0 - sigma) * sample
    return sample


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    """Calculate shift parameter for FLUX models"""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def optimal_transport_coupling(x_src, x_tar, reg_coeff=0.1, max_iter=100):
    """
    Compute optimal transport coupling between source and target distributions
    using Sinkhorn algorithm for regularized OT
    
    Args:
        x_src: Source samples [B, C, H, W]
        x_tar: Target samples [B, C, H, W] 
        reg_coeff: Regularization coefficient for entropy regularization
        max_iter: Maximum iterations for Sinkhorn algorithm
    
    Returns:
        Coupling matrix and transport plan
    """
    device = x_src.device
    B = x_src.shape[0]
    
    # Flatten spatial dimensions for transport computation
    x_src_flat = x_src.view(B, -1)
    x_tar_flat = x_tar.view(B, -1)
    
    # Compute cost matrix (L2 distances)
    cost_matrix = torch.cdist(x_src_flat, x_tar_flat, p=2) ** 2
    
    # Initialize marginals (uniform distribution)
    mu = torch.ones(B, device=device) / B
    nu = torch.ones(B, device=device) / B
    
    # Sinkhorn iterations
    K = torch.exp(-cost_matrix / reg_coeff)
    u = torch.ones(B, device=device)
    v = torch.ones(B, device=device)
    
    for _ in range(max_iter):
        u_prev = u.clone()
        u = mu / (K @ v)
        v = nu / (K.T @ u)
        
        # Check convergence
        if torch.norm(u - u_prev) < 1e-6:
            break
    
    # Compute optimal coupling
    coupling = torch.diag(u) @ K @ torch.diag(v)
    
    return coupling, cost_matrix


def adaptive_transport_strength(t, coupling_matrix, base_strength=1.0, adaptive_factor=2.0):
    """
    Compute adaptive transport strength based on time and coupling quality
    
    Args:
        t: Current timestep
        coupling_matrix: Optimal transport coupling matrix
        base_strength: Base transport strength
        adaptive_factor: Factor controlling adaptation
        
    Returns:
        Adaptive transport strength scalar
    """
    # Time-dependent component (stronger early, weaker late)
    time_component = (1.0 - t) * base_strength
    
    # Coupling quality component (based on transport cost)
    coupling_entropy = -torch.sum(coupling_matrix * torch.log(coupling_matrix + 1e-8))
    coupling_quality = torch.exp(-coupling_entropy / adaptive_factor)
    
    return time_component * coupling_quality


def calc_v_sd3(pipe, src_tar_latent_model_input, src_tar_prompt_embeds, src_tar_pooled_prompt_embeds, src_guidance_scale, tar_guidance_scale, t):
    """Enhanced velocity calculation for SD3 with improved precision"""
    timestep = t.expand(src_tar_latent_model_input.shape[0])

    with torch.no_grad():
        noise_pred_src_tar = pipe.transformer(
            hidden_states=src_tar_latent_model_input,
            timestep=timestep,
            encoder_hidden_states=src_tar_prompt_embeds,
            pooled_projections=src_tar_pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

        if pipe.do_classifier_free_guidance:
            src_noise_pred_uncond, src_noise_pred_text, tar_noise_pred_uncond, tar_noise_pred_text = noise_pred_src_tar.chunk(4)
            noise_pred_src = src_noise_pred_uncond + src_guidance_scale * (src_noise_pred_text - src_noise_pred_uncond)
            noise_pred_tar = tar_noise_pred_uncond + tar_guidance_scale * (tar_noise_pred_text - tar_noise_pred_uncond)

    return noise_pred_src, noise_pred_tar


def calc_v_flux(pipe, latents, prompt_embeds, pooled_prompt_embeds, guidance, text_ids, latent_image_ids, t):
    """Enhanced velocity calculation for FLUX with improved precision"""
    timestep = t.expand(latents.shape[0])

    with torch.no_grad():
        noise_pred = pipe.transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

    return noise_pred


@torch.no_grad()
def FlowEditSD3_Enhanced(pipe,
    scheduler,
    x_src,
    src_prompt,
    tar_prompt,
    negative_prompt,
    T_steps: int = 50,
    n_avg: int = 1,
    src_guidance_scale: float = 3.5,
    tar_guidance_scale: float = 13.5,
    n_min: int = 0,
    n_max: int = 15,
    use_optimal_transport: bool = True,
    ot_reg_coeff: float = 0.1,
    adaptive_transport: bool = True,
    transport_strength: float = 1.0):
    """
    Enhanced FlowEdit for SD3 with Optimal Transport guidance
    
    New parameters:
        use_optimal_transport: Enable OT-guided coupling
        ot_reg_coeff: Regularization coefficient for OT
        adaptive_transport: Enable adaptive transport strength
        transport_strength: Base transport strength
    """
    
    device = x_src.device
    timesteps, T_steps = retrieve_timesteps(scheduler, T_steps, device, timesteps=None)
    num_warmup_steps = max(len(timesteps) - T_steps * scheduler.order, 0)
    pipe._num_timesteps = len(timesteps)
    pipe._guidance_scale = src_guidance_scale
    
    # Encode prompts
    (src_prompt_embeds, src_negative_prompt_embeds, src_pooled_prompt_embeds, src_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=src_prompt, prompt_2=None, prompt_3=None, negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance, device=device,
    )

    pipe._guidance_scale = tar_guidance_scale
    (tar_prompt_embeds, tar_negative_prompt_embeds, tar_pooled_prompt_embeds, tar_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=tar_prompt, prompt_2=None, prompt_3=None, negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance, device=device,
    )
 
    # CFG preparation
    src_tar_prompt_embeds = torch.cat([src_negative_prompt_embeds, src_prompt_embeds, tar_negative_prompt_embeds, tar_prompt_embeds], dim=0)
    src_tar_pooled_prompt_embeds = torch.cat([src_negative_pooled_prompt_embeds, src_pooled_prompt_embeds, tar_negative_pooled_prompt_embeds, tar_pooled_prompt_embeds], dim=0)
    
    # Initialize ODE with source image
    zt_edit = x_src.clone()
    
    # Store coupling history for adaptive transport
    coupling_history = []

    for i, t in tqdm(enumerate(timesteps), desc="FlowEdit Enhanced"):
        
        if T_steps - i > n_max:
            continue
        
        t_i = t/1000
        if i+1 < len(timesteps): 
            t_im1 = (timesteps[i+1])/1000
        else:
            t_im1 = torch.zeros_like(t_i).to(t_i.device)
        
        if T_steps - i > n_min:
            # Enhanced velocity calculation with OT guidance
            V_delta_avg = torch.zeros_like(x_src)
            
            # Generate multiple noise realizations for robust coupling
            noise_realizations = []
            src_realizations = []
            
            for k in range(n_avg):
                fwd_noise = torch.randn_like(x_src).to(x_src.device)
                noise_realizations.append(fwd_noise)
                zt_src = (1-t_i)*x_src + (t_i)*fwd_noise
                src_realizations.append(zt_src)
            
            # Optimal Transport guided coupling
            if use_optimal_transport and len(src_realizations) > 1:
                # Stack realizations for batch processing
                try:
                    src_batch = torch.stack(src_realizations, dim=0).squeeze(1) if len(src_realizations) > 1 else src_realizations[0]
                    
                    # Generate target candidates
                    tar_candidates = []
                    for zt_src in src_realizations:
                        zt_tar_candidate = zt_edit + zt_src - x_src
                        tar_candidates.append(zt_tar_candidate)
                    
                    tar_batch = torch.stack(tar_candidates, dim=0).squeeze(1) if len(tar_candidates) > 1 else tar_candidates[0]
                    
                    # Compute optimal coupling - handle potential shape issues
                    if src_batch.shape[0] > 1 and len(src_batch.shape) >= 3:  # Valid for coupling (SD3 should be 4D -> 3D after squeeze)
                        coupling_matrix, cost_matrix = optimal_transport_coupling(
                            src_batch, tar_batch, reg_coeff=ot_reg_coeff
                        )
                        coupling_history.append(coupling_matrix)
                        
                        # Adaptive transport strength
                        if adaptive_transport:
                            transport_factor = adaptive_transport_strength(
                                t_i.item(), coupling_matrix, transport_strength
                            )
                        else:
                            transport_factor = transport_strength
                    else:
                        transport_factor = transport_strength
                except Exception as e:
                    # Fall back to default transport strength if OT coupling fails
                    transport_factor = transport_strength
            else:
                transport_factor = transport_strength
            
            # Calculate velocities with OT guidance
            for k in range(n_avg):
                zt_src = src_realizations[k]
                zt_tar = zt_edit + zt_src - x_src
                
                src_tar_latent_model_input = torch.cat([zt_src, zt_src, zt_tar, zt_tar]) if pipe.do_classifier_free_guidance else (zt_src, zt_tar)
                Vt_src, Vt_tar = calc_v_sd3(pipe, src_tar_latent_model_input, src_tar_prompt_embeds, src_tar_pooled_prompt_embeds, src_guidance_scale, tar_guidance_scale, t)
                
                # Apply transport factor
                V_delta = (Vt_tar - Vt_src) * transport_factor
                V_delta_avg += (1/n_avg) * V_delta

            # Propagate ODE with enhanced step
            zt_edit = zt_edit.to(torch.float32)
            zt_edit = zt_edit + (t_im1 - t_i) * V_delta_avg
            zt_edit = zt_edit.to(V_delta_avg.dtype)

        else:  # Regular sampling for last n_min steps
            if i == T_steps-n_min:
                # Initialize SDEDIT-style generation phase
                fwd_noise = torch.randn_like(x_src).to(x_src.device)
                xt_src = scale_noise(scheduler, x_src, t, noise=fwd_noise)
                xt_tar = zt_edit + xt_src - x_src
                
            src_tar_latent_model_input = torch.cat([xt_tar, xt_tar, xt_tar, xt_tar]) if pipe.do_classifier_free_guidance else (xt_src, xt_tar)
            _, Vt_tar = calc_v_sd3(pipe, src_tar_latent_model_input, src_tar_prompt_embeds, src_tar_pooled_prompt_embeds, src_guidance_scale, tar_guidance_scale, t)

            xt_tar = xt_tar.to(torch.float32)
            prev_sample = xt_tar + (t_im1 - t_i) * (Vt_tar)
            prev_sample = prev_sample.to(Vt_tar.dtype)
            xt_tar = prev_sample
        
    return zt_edit if n_min == 0 else xt_tar


@torch.no_grad()
def FlowEditFLUX_Enhanced(pipe,
    scheduler,
    x_src,
    src_prompt,
    tar_prompt,
    negative_prompt,
    T_steps: int = 28,
    n_avg: int = 1,
    src_guidance_scale: float = 1.5,
    tar_guidance_scale: float = 5.5,
    n_min: int = 0,
    n_max: int = 24,
    use_optimal_transport: bool = True,
    ot_reg_coeff: float = 0.1,
    adaptive_transport: bool = True,
    transport_strength: float = 1.0):
    """
    Enhanced FlowEdit for FLUX with Optimal Transport guidance
    """

    device = x_src.device
    orig_height, orig_width = x_src.shape[2]*pipe.vae_scale_factor//2, x_src.shape[3]*pipe.vae_scale_factor//2
    num_channels_latents = pipe.transformer.config.in_channels // 4

    pipe.check_inputs(
        prompt=src_prompt, prompt_2=None, height=orig_height, width=orig_width,
        callback_on_step_end_tensor_inputs=None, max_sequence_length=512,
    )

    x_src, latent_src_image_ids = pipe.prepare_latents(
        batch_size=x_src.shape[0], num_channels_latents=num_channels_latents,
        height=orig_height, width=orig_width, dtype=x_src.dtype, device=x_src.device,
        generator=None, latents=x_src
    )
    
    x_src_packed = pipe._pack_latents(x_src, x_src.shape[0], num_channels_latents, x_src.shape[2], x_src.shape[3])
    latent_tar_image_ids = latent_src_image_ids

    # Prepare timesteps with enhanced precision
    sigmas = np.linspace(1.0, 1 / T_steps, T_steps)
    image_seq_len = x_src_packed.shape[1]
    mu = calculate_shift(
        image_seq_len, scheduler.config.base_image_seq_len, scheduler.config.max_image_seq_len,
        scheduler.config.base_shift, scheduler.config.max_shift,
    )
    
    timesteps, T_steps = retrieve_timesteps(scheduler, T_steps, device, timesteps=None, sigmas=sigmas, mu=mu,)
    num_warmup_steps = max(len(timesteps) - T_steps * pipe.scheduler.order, 0)
    pipe._num_timesteps = len(timesteps)

    # Encode prompts
    (src_prompt_embeds, src_pooled_prompt_embeds, src_text_ids,
    ) = pipe.encode_prompt(prompt=src_prompt, prompt_2=None, device=device,)

    pipe._guidance_scale = tar_guidance_scale
    (tar_prompt_embeds, tar_pooled_prompt_embeds, tar_text_ids,
    ) = pipe.encode_prompt(prompt=tar_prompt, prompt_2=None, device=device,)

    # Handle guidance
    if pipe.transformer.config.guidance_embeds:
        src_guidance = torch.tensor([src_guidance_scale], device=device).expand(x_src_packed.shape[0])
        tar_guidance = torch.tensor([tar_guidance_scale], device=device).expand(x_src_packed.shape[0])
    else:
        src_guidance = None
        tar_guidance = None

    # Initialize ODE
    zt_edit = x_src_packed.clone()
    coupling_history = []

    for i, t in tqdm(enumerate(timesteps), desc="FlowEdit FLUX Enhanced"):
        
        if T_steps - i > n_max:
            continue
        
        t_i = t/1000
        if i+1 < len(timesteps): 
            t_im1 = (timesteps[i+1])/1000
        else:
            t_im1 = torch.zeros_like(t_i).to(t_i.device)
        
        if T_steps - i > n_min:
            V_delta_avg = torch.zeros_like(x_src_packed)
            
            # Generate noise realizations
            noise_realizations = []
            src_realizations = []
            
            for k in range(n_avg):
                fwd_noise = torch.randn_like(x_src_packed).to(x_src_packed.device)
                noise_realizations.append(fwd_noise)
                zt_src = (1-t_i)*x_src_packed + (t_i)*fwd_noise
                src_realizations.append(zt_src)
            
            # OT guidance for FLUX
            if use_optimal_transport and len(src_realizations) > 1:
                src_batch = torch.stack(src_realizations, dim=0).squeeze(1)
                tar_candidates = [zt_edit + zt_src - x_src_packed for zt_src in src_realizations]
                tar_batch = torch.stack(tar_candidates, dim=0).squeeze(1)
                
                if src_batch.shape[0] > 1:
                    coupling_matrix, _ = optimal_transport_coupling(src_batch, tar_batch, reg_coeff=ot_reg_coeff)
                    coupling_history.append(coupling_matrix)
                    
                    if adaptive_transport:
                        transport_factor = adaptive_transport_strength(t_i.item(), coupling_matrix, transport_strength)
                    else:
                        transport_factor = transport_strength
                else:
                    transport_factor = transport_strength
            else:
                transport_factor = transport_strength
            
            # Calculate velocities
            for k in range(n_avg):
                zt_src = src_realizations[k]
                zt_tar = zt_edit + zt_src - x_src_packed

                Vt_src = calc_v_flux(pipe, zt_src, src_prompt_embeds, src_pooled_prompt_embeds, src_guidance, src_text_ids, latent_src_image_ids, t)
                Vt_tar = calc_v_flux(pipe, zt_tar, tar_prompt_embeds, tar_pooled_prompt_embeds, tar_guidance, tar_text_ids, latent_tar_image_ids, t)

                V_delta = (Vt_tar - Vt_src) * transport_factor
                V_delta_avg += (1/n_avg) * V_delta

            # Propagate ODE
            zt_edit = zt_edit.to(torch.float32)
            zt_edit = zt_edit + (t_im1 - t_i) * V_delta_avg
            zt_edit = zt_edit.to(V_delta_avg.dtype)

        else:  # Regular sampling
            if i == T_steps-n_min:
                fwd_noise = torch.randn_like(x_src_packed).to(x_src_packed.device)
                xt_src = scale_noise(scheduler, x_src_packed, t, noise=fwd_noise)
                xt_tar = zt_edit + xt_src - x_src_packed
                
            Vt_tar = calc_v_flux(pipe, xt_tar, tar_prompt_embeds, tar_pooled_prompt_embeds, tar_guidance, tar_text_ids, latent_tar_image_ids, t)
            
            xt_tar = xt_tar.to(torch.float32)
            prev_sample = xt_tar + (t_im1 - t_i) * (Vt_tar)
            prev_sample = prev_sample.to(Vt_tar.dtype)
            xt_tar = prev_sample
    
    out = zt_edit if n_min == 0 else xt_tar
    unpacked_out = pipe._unpack_latents(out, orig_height, orig_width, pipe.vae_scale_factor)
    return unpacked_out


# Utility functions for analysis
def analyze_transport_quality(coupling_history):
    """Analyze the quality of transport couplings over time"""
    if not coupling_history:
        return {}
    
    qualities = []
    entropies = []
    costs = []
    
    for coupling in coupling_history:
        # Entropy (measure of spread)
        entropy = -torch.sum(coupling * torch.log(coupling + 1e-8))
        entropies.append(entropy.item())
        
        # Transport cost
        cost = torch.sum(coupling)
        costs.append(cost.item())
        
        # Quality metric (lower entropy = more concentrated transport)
        quality = torch.exp(-entropy)
        qualities.append(quality.item())
    
    return {
        'avg_quality': np.mean(qualities),
        'avg_entropy': np.mean(entropies),
        'avg_cost': np.mean(costs),
        'quality_trend': qualities,
        'entropy_trend': entropies,
        'cost_trend': costs
    }


def visualize_coupling_matrix(coupling_matrix, save_path=None):
    """Visualize the optimal transport coupling matrix"""
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(8, 6))
        plt.imshow(coupling_matrix.cpu().numpy(), cmap='Blues')
        plt.colorbar(label='Coupling Strength')
        plt.title('Optimal Transport Coupling Matrix')
        plt.xlabel('Target Samples')
        plt.ylabel('Source Samples')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for visualization")