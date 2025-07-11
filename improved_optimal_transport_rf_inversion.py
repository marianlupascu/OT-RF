import torch
import torch.nn.functional as F
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps

# Import the base class without modification
try:
    from pipeline_rf_inversion_sde import RFInversionFluxPipelineSDE as BaseClass
except ImportError:
    # Fall back to using RFInversionFluxPipeline as the base class
    try:
        from pipeline_flux_rf_inversion import RFInversionFluxPipeline as BaseClass
    except ImportError:
        # If we can't import directly, let's assume we're using the class from diffusers
        from diffusers import DiffusionPipeline
        BaseClass = DiffusionPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.float16,
            custom_pipeline="pipeline_flux_rf_inversion"
        ).__class__


class OptimalTransportRFInversionPipeline(BaseClass):
    """
    Enhanced RF-Inversion pipeline with Optimal Transport principles for improved
    image editing with better artifact prevention and prompt alignment.
    """
    
    @classmethod
    def from_pipe(cls, pipe):
        """
        Create an instance of this class from an existing pipeline.
        
        Args:
            pipe: An existing RF-Inversion pipeline
            
        Returns:
            An instance of OptimalTransportRFInversionPipeline
        """
        if not hasattr(pipe, 'invert'):
            raise ValueError("The provided pipeline must have an 'invert' method")
        
        # Create a new instance with the same components
        instance = cls(
            scheduler=pipe.scheduler,
            vae=pipe.vae,
            text_encoder=pipe.text_encoder,
            tokenizer=pipe.tokenizer,
            text_encoder_2=pipe.text_encoder_2,
            tokenizer_2=pipe.tokenizer_2,
            transformer=pipe.transformer,
        )
        
        # Copy any additional attributes
        for attr_name in dir(pipe):
            if not attr_name.startswith('__') and not hasattr(instance, attr_name):
                try:
                    setattr(instance, attr_name, getattr(pipe, attr_name))
                except AttributeError:
                    pass
                    
        return instance

    def compute_ot_direction(
        self, 
        current_state: torch.Tensor, 
        target_state: torch.Tensor, 
        timestep: float,
        smooth: bool = True,
    ) -> torch.Tensor:
        """
        Compute the optimal transport direction from current state to target state.
        
        Args:
            current_state: Current latent state tensor
            target_state: Target latent state tensor
            timestep: Current timestep in [0,1]
            smooth: Whether to apply smoothing to the direction
            
        Returns:
            OT direction tensor
        """
        # Simple OT direction is just the normalized vector from current to target
        # For Wasserstein-2 with quadratic cost, this is exactly the gradient
        direction = (target_state - current_state) / max(1.0 - timestep, 0.01)
        
        # Prevent extreme values that could cause artifacts
        max_norm = 10.0
        direction_norm = torch.norm(direction, dim=-1, keepdim=True)
        scale_factor = torch.minimum(
            torch.ones_like(direction_norm),
            max_norm / (direction_norm + 1e-8)
        )
        direction = direction * scale_factor
        
        return direction

    def get_adaptive_transport_strength(
        self,
        step_index: int,
        total_steps: int,
        base_strength: float,
        timestep_value: float,
        phase_shift: float = 0.3
    ) -> float:
        """
        Calculate transport strength that adapts throughout the denoising process.
        
        Args:
            step_index: Current denoising step index
            total_steps: Total number of denoising steps
            base_strength: Base transport strength parameter
            timestep_value: Current timestep value
            phase_shift: Controls the falloff curve
            
        Returns:
            Adapted transport strength
        """
        # Normalize step index to [0, 1]
        relative_step = step_index / total_steps
        
        # Cosine falloff - starts at full strength and gradually decreases
        # This provides a smooth transition from OT-guided to prompt-guided denoising
        falloff = 0.5 * (1.0 + np.cos(min(relative_step / phase_shift, 1.0) * np.pi))
        
        # Further reduce strength in very early steps (helps with initialization)
        if relative_step < 0.05:
            early_ramp = relative_step / 0.05
            falloff = falloff * early_ramp
            
        return base_strength * falloff

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        inverted_latents: Optional[torch.FloatTensor] = None,
        image_latents: Optional[torch.FloatTensor] = None,
        latent_image_ids: Optional[torch.FloatTensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        eta: float = 1.0,
        decay_eta: Optional[bool] = False,
        eta_decay_power: Optional[float] = 1.0,
        strength: float = 1.0,
        start_timestep: float = 0,
        stop_timestep: float = 0.25,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        timesteps: List[int] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        
        # OT specific parameters
        use_ot_guidance: bool = True,
        transport_strength: float = 0.1,
        ot_guidance_phase: float = 0.3,
        edit_mask: Optional[torch.FloatTensor] = None,
        use_regional_guidance: bool = False,
        respect_prompt_guidance: bool = True,
    ):
        """
        Enhanced RF-Inversion pipeline that incorporates optimal transport guidance
        for better editing results with fewer artifacts.
        
        Args:
            [All parameters from the base class...]
            
            use_ot_guidance (bool, optional, defaults to True):
                Whether to use optimal transport guidance during sampling.
            transport_strength (float, optional, defaults to 0.1):
                Strength of the optimal transport guidance.
            ot_guidance_phase (float, optional, defaults to 0.3):
                Fraction of denoising steps to apply OT guidance.
            edit_mask (torch.FloatTensor, optional):
                Mask for regional guidance (1=apply OT, 0=skip OT).
            use_regional_guidance (bool, optional, defaults to False):
                Whether to use the provided mask for regional editing.
            respect_prompt_guidance (bool, optional, defaults to True):
                Whether to prioritize prompt guidance over OT guidance.
                
        Returns:
            FluxPipelineOutput: The generated images.
        """
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Input validation
        self.check_inputs(
            prompt,
            prompt_2,
            inverted_latents,
            image_latents,
            latent_image_ids,
            height,
            width,
            start_timestep,
            stop_timestep,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False
        do_rf_inversion = inverted_latents is not None

        # 2. Prepare parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode prompt
        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        if do_rf_inversion:
            latents = inverted_latents
        else:
            latents, latent_image_ids = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = (int(height) // self.vae_scale_factor // 2) * (int(width) // self.vae_scale_factor // 2)
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            mu=mu,
        )
        if do_rf_inversion:
            start_timestep_idx = int(start_timestep * num_inference_steps)
            stop_timestep_idx = min(int(stop_timestep * num_inference_steps), num_inference_steps)
            timesteps, sigmas, num_inference_steps = self.get_timesteps(num_inference_steps, strength)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # Handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        # Process edit mask if provided
        if use_regional_guidance and edit_mask is not None:
            # Ensure mask is in correct format
            if edit_mask.dim() != 4:
                raise ValueError("Edit mask must be 4D tensor [B, 1, H, W]")
                
            # Resize mask to match latent dimensions
            latent_h = int(height // self.vae_scale_factor // 2)
            latent_w = int(width // self.vae_scale_factor // 2)
            
            edit_mask = F.interpolate(
                edit_mask,
                size=(latent_h, latent_w),
                mode='nearest'
            )
            # Reshape mask to match latent sequence structure (B, L, 1)
            edit_mask = edit_mask.reshape(batch_size, 1, -1).transpose(1, 2)
        else:
            edit_mask = None

        if do_rf_inversion:
            y_0 = image_latents.clone()

        # 6. Denoising loop with OT guidance
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if do_rf_inversion:
                    # ti (current timestep) as annotated in algorithm 2 - i/num_inference_steps
                    t_i = 1 - t / 1000

                if self.interrupt:
                    continue

                # Broadcast to batch dimension 
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                # Get model prediction
                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # Process the prediction with RF-Inversion approach
                if do_rf_inversion:
                    # Calculate velocity field
                    v_t = -noise_pred
                    
                    # Calculate conditional velocity field (towards original image)
                    v_t_cond = (y_0 - latents) / (1 - t_i)
                    
                    # Determine if we should apply OT guidance at this step
                    apply_ot = use_ot_guidance and (i / len(timesteps)) < ot_guidance_phase
                    
                    # Apply OT guidance if enabled
                    if apply_ot:
                        # Get adaptive strength based on current step
                        ot_strength = self.get_adaptive_transport_strength(
                            i, len(timesteps), transport_strength, t_i, ot_guidance_phase
                        )
                        
                        # Calculate OT direction
                        ot_direction = self.compute_ot_direction(latents, y_0, t_i)
                        
                        # Apply regional guidance if enabled
                        if use_regional_guidance and edit_mask is not None:
                            # Expand mask to broadcast properly
                            mask = edit_mask.expand_as(v_t)
                            # Apply mask to OT direction
                            ot_contribution = ot_strength * mask * (ot_direction - v_t)
                        else:
                            # Global guidance
                            ot_contribution = ot_strength * (ot_direction - v_t)
                        
                        # Modify velocity with OT guidance, respecting prompt guidance
                        if respect_prompt_guidance:
                            # Apply OT guidance while preserving model's prediction direction
                            v_t_norm = torch.norm(v_t, dim=-1, keepdim=True)
                            ot_direction_norm = torch.norm(ot_direction, dim=-1, keepdim=True)
                            
                            # Normalize the OT contribution to keep magnitude similar to v_t
                            scale_factor = v_t_norm / (ot_direction_norm + 1e-8)
                            ot_contribution = ot_contribution * scale_factor
                            
                            # Add contribution to v_t
                            v_t = v_t + ot_contribution
                        else:
                            # Simpler blending
                            v_t = v_t + ot_contribution
                    
                    # Apply eta guidance (controlling editability vs. faithfulness)
                    eta_t = eta if start_timestep_idx <= i < stop_timestep_idx else 0.0
                    if decay_eta:
                        eta_t = eta_t * (1 - i / num_inference_steps) ** eta_decay_power
                        
                    # Blend between v_t and v_t_cond based on eta
                    v_hat_t = v_t + eta_t * (v_t_cond - v_t)
                    
                    # Update latents
                    latents = latents + v_hat_t * (sigmas[i] - sigmas[i + 1])
                else:
                    # Standard diffusion process
                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                # Handle datatype conversion
                latents_dtype = latents.dtype
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # Handle MPS quirk
                        latents = latents.to(latents_dtype)

                # Handle callbacks
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # Update progress
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # Process the final result
        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)

