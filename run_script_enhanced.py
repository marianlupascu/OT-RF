#!/usr/bin/env python3
"""
Enhanced FlowEdit Script with Optimal Transport Support
Usage: python run_script_enhanced.py --exp_yaml FLUX_exp_enhanced.yaml
"""

import torch
from diffusers import StableDiffusion3Pipeline
from diffusers import FluxPipeline
from PIL import Image
import argparse
import random 
import numpy as np
import yaml
import os
import time
import json
from datetime import datetime

# Import both original and enhanced versions
from FlowEdit_utils import FlowEditSD3, FlowEditFLUX
from FlowEdit_utils_enhanced import (
    FlowEditSD3_Enhanced, 
    FlowEditFLUX_Enhanced
)


def save_enhanced_metrics(transport_analysis, save_dir, filename="transport_metrics.json"):
    """Save transport analysis metrics to JSON file"""
    if not transport_analysis:
        return
    
    metrics_path = os.path.join(save_dir, filename)
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_analysis = {}
    for key, value in transport_analysis.items():
        if isinstance(value, (list, np.ndarray)):
            serializable_analysis[key] = list(value) if hasattr(value, '__iter__') else value
        elif isinstance(value, torch.Tensor):
            serializable_analysis[key] = value.item()
        else:
            serializable_analysis[key] = float(value) if isinstance(value, (int, float, np.number)) else value
    
    with open(metrics_path, 'w') as f:
        json.dump(serializable_analysis, f, indent=2)
    
    print(f"📊 Transport metrics saved to: {metrics_path}")


def create_comparison_grid(images_dict, save_path, titles=None):
    """Create a comparison grid of images"""
    from PIL import Image, ImageDraw, ImageFont
    
    images = list(images_dict.values())
    if not images:
        return
    
    # Calculate grid dimensions
    n_images = len(images)
    cols = min(4, n_images)
    rows = (n_images + cols - 1) // cols
    
    # Get image size (assume all images are same size)
    img_width, img_height = images[0].size
    
    # Create grid
    grid_width = cols * img_width
    grid_height = rows * img_height + 40 * rows  # Extra space for titles
    
    grid_image = Image.new('RGB', (grid_width, grid_height), 'white')
    
    # Add images to grid
    for i, (key, img) in enumerate(images_dict.items()):
        row = i // cols
        col = i % cols
        
        x = col * img_width
        y = row * (img_height + 40) + 40
        
        grid_image.paste(img, (x, y))
        
        # Add title
        try:
            draw = ImageDraw.Draw(grid_image)
            title = titles[i] if titles and i < len(titles) else key
            draw.text((x + 10, y - 35), title, fill='black')
        except:
            pass  # Skip if font not available
    
    grid_image.save(save_path)
    print(f"🖼️ Comparison grid saved to: {save_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Enhanced FlowEdit with Optimal Transport")
    parser.add_argument("--device_number", type=int, default=0, help="device number to use")
    parser.add_argument("--exp_yaml", type=str, default="FLUX_exp_enhanced.yaml", help="experiment yaml file")
    parser.add_argument("--compare_methods", action="store_true", 
                        help="Run both original and enhanced methods for comparison")
    parser.add_argument("--save_transport_analysis", action="store_true",
                        help="Save detailed transport analysis")
    parser.add_argument("--create_comparison_grids", action="store_true",
                        help="Create comparison grids for visual analysis")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")

    args = parser.parse_args()

    # Set device
    device_number = args.device_number
    device = torch.device(f"cuda:{device_number}" if torch.cuda.is_available() else "cpu")
    
    print("🚀 Enhanced FlowEdit with Optimal Transport")
    print("=" * 50)
    print(f"🔧 Device: {device}")
    if torch.cuda.is_available():
        print(f"💾 GPU: {torch.cuda.get_device_name()}")
        print(f"🔥 CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load experiment configuration
    exp_yaml = args.exp_yaml
    print(f"📋 Loading experiment config: {exp_yaml}")
    
    try:
        with open(exp_yaml) as file:
            exp_configs = yaml.load(file, Loader=yaml.FullLoader)
    except FileNotFoundError:
        print(f"❌ Error: Configuration file not found: {exp_yaml}")
        print("💡 Please ensure the YAML file exists in the current directory")
        exit(1)
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        exit(1)

    if not exp_configs:
        print("❌ Error: Empty configuration file")
        exit(1)

    model_type = exp_configs[0]["model_type"]  # Currently only one model type per run
    print(f"🤖 Model type: {model_type}")

    # Load pipeline
    print(f"⚡ Loading {model_type} pipeline...")
    try:
        if model_type == 'FLUX':
            pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16)
        elif model_type == 'SD3':
            pipe = StableDiffusion3Pipeline.from_pretrained(
                "stabilityai/stable-diffusion-3-medium-diffusers", 
                torch_dtype=torch.float16
            )
        else:
            raise NotImplementedError(f"Model type {model_type} not implemented")
        
        scheduler = pipe.scheduler
        pipe = pipe.to(device)
        print("✅ Pipeline loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading pipeline: {e}")
        print("💡 Make sure you have access to the model and sufficient memory")
        exit(1)

    # Process each experiment
    for exp_idx, exp_dict in enumerate(exp_configs):
        
        print(f"\n🎯 Processing experiment {exp_idx + 1}/{len(exp_configs)}")
        print("-" * 40)
        
        # Extract experiment parameters
        exp_name = exp_dict["exp_name"]
        T_steps = exp_dict["T_steps"]
        n_avg = exp_dict["n_avg"]
        src_guidance_scale = exp_dict["src_guidance_scale"]
        tar_guidance_scale = exp_dict["tar_guidance_scale"]
        n_min = exp_dict["n_min"]
        n_max = exp_dict["n_max"]
        seed = exp_dict["seed"]
        
        # Enhanced parameters (with defaults if not present)
        use_optimal_transport = exp_dict.get("use_optimal_transport", True)
        ot_reg_coeff = exp_dict.get("ot_reg_coeff", 0.1)
        adaptive_transport = exp_dict.get("adaptive_transport", True)
        transport_strength = exp_dict.get("transport_strength", 1.0)
        
        print(f"📝 Experiment: {exp_name}")
        print(f"🎲 Seed: {seed}")
        print(f"🔄 Enhanced OT: {use_optimal_transport}")
        
        if args.verbose:
            print(f"📊 Parameters:")
            print(f"   T_steps: {T_steps}, n_avg: {n_avg}")
            print(f"   Source guidance: {src_guidance_scale}")
            print(f"   Target guidance: {tar_guidance_scale}")
            print(f"   n_min: {n_min}, n_max: {n_max}")
            print(f"   OT reg coeff: {ot_reg_coeff}")
            print(f"   Transport strength: {transport_strength}")
            print(f"   Adaptive transport: {adaptive_transport}")
        
        # Set seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Load dataset configuration
        dataset_yaml = exp_dict["dataset_yaml"]
        try:
            with open(dataset_yaml) as file:
                dataset_configs = yaml.load(file, Loader=yaml.FullLoader)
        except FileNotFoundError:
            print(f"❌ Error: Dataset file not found: {dataset_yaml}")
            continue
        except Exception as e:
            print(f"❌ Error loading dataset config: {e}")
            continue

        # Process each dataset item
        for data_idx, data_dict in enumerate(dataset_configs):
            
            src_prompt = data_dict["source_prompt"]
            tar_prompts = data_dict["target_prompts"]
            negative_prompt = ""  # Can be extended for SD3
            image_src_path = data_dict["input_img"]

            print(f"\n📷 Processing image: {os.path.basename(image_src_path)}")
            print(f"🎨 Source prompt: {src_prompt}")

            # Load and preprocess image
            try:
                if not os.path.exists(image_src_path):
                    print(f"⚠️ Warning: Image not found: {image_src_path}")
                    continue
                    
                image = Image.open(image_src_path).convert('RGB')
                # Crop to be divisible by 16
                image = image.crop((0, 0, image.width - image.width % 16, image.height - image.height % 16))
                image_src = pipe.image_processor.preprocess(image)
                image_src = image_src.to(device).half()
                
                # Encode to latent space
                with torch.autocast("cuda"), torch.inference_mode():
                    x0_src_denorm = pipe.vae.encode(image_src).latent_dist.mode()
                x0_src = (x0_src_denorm - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
                x0_src = x0_src.to(device)
                
                if args.verbose:
                    print(f"🔍 Image processed: {image.size} -> latent {x0_src.shape}")
                
            except Exception as e:
                print(f"❌ Error loading image {image_src_path}: {e}")
                continue
            
            # Process each target prompt
            for tar_num, tar_prompt in enumerate(tar_prompts):
                
                print(f"🎯 Target {tar_num + 1}: {tar_prompt}")
                
                start_time = time.time()
                
                # Prepare enhanced parameters
                enhanced_params = {
                    "T_steps": T_steps,
                    "n_avg": n_avg,
                    "src_guidance_scale": src_guidance_scale,
                    "tar_guidance_scale": tar_guidance_scale,
                    "n_min": n_min,
                    "n_max": n_max,
                    "use_optimal_transport": use_optimal_transport,
                    "ot_reg_coeff": ot_reg_coeff,
                    "adaptive_transport": adaptive_transport,
                    "transport_strength": transport_strength
                }
                
                try:
                    # Run enhanced FlowEdit
                    if model_type == 'SD3':
                        x0_tar = FlowEditSD3_Enhanced(
                            pipe=pipe,
                            scheduler=scheduler,
                            x_src=x0_src,
                            src_prompt=src_prompt,
                            tar_prompt=tar_prompt,
                            negative_prompt=negative_prompt,
                            **enhanced_params
                        )
                        
                    elif model_type == 'FLUX':
                        x0_tar = FlowEditFLUX_Enhanced(
                            pipe=pipe,
                            scheduler=scheduler,
                            x_src=x0_src,
                            src_prompt=src_prompt,
                            tar_prompt=tar_prompt,
                            negative_prompt=negative_prompt,
                            **enhanced_params
                        )
                    else:
                        raise NotImplementedError(f"Model type {model_type} not implemented")

                    editing_time = time.time() - start_time
                    print(f"⏱️ Editing completed in {editing_time:.2f}s")

                    # Decode to image space
                    x0_tar_denorm = (x0_tar / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
                    with torch.autocast("cuda"), torch.inference_mode():
                        image_tar = pipe.vae.decode(x0_tar_denorm, return_dict=False)[0]
                    image_tar = pipe.image_processor.postprocess(image_tar)

                    # Prepare save directories and filenames
                    src_prompt_txt = data_dict["input_img"].split("/")[-1].split(".")[0]
                    tar_prompt_txt = str(tar_num)
                    
                    base_save_dir = f"outputs/{exp_name}/{model_type}/src_{src_prompt_txt}/tar_{tar_prompt_txt}"
                    enhanced_save_dir = f"{base_save_dir}/enhanced"
                    os.makedirs(enhanced_save_dir, exist_ok=True)
                    
                    # Enhanced filename with OT parameters
                    ot_suffix = f"_OT{ot_reg_coeff:.3f}_TS{transport_strength:.1f}_AT{adaptive_transport}"
                    enhanced_filename = (f"enhanced_T{T_steps}_avg{n_avg}_"
                                       f"cfg{src_guidance_scale}_{tar_guidance_scale}_"
                                       f"n{n_min}_{n_max}{ot_suffix}_seed{seed}.png")
                    
                    enhanced_path = os.path.join(enhanced_save_dir, enhanced_filename)
                    image_tar[0].save(enhanced_path)
                    
                    print(f"💾 Enhanced result saved: {enhanced_filename}")
                    
                    # Transport analysis
                    transport_analysis = {}
                    if args.save_transport_analysis and use_optimal_transport:
                        print("🔬 Performing transport analysis...")
                        try:
                            # Import here to avoid issues if not available
                            from FlowEdit_utils_enhanced import OptimalTransportFlowEdit
                            ot_guidance = OptimalTransportFlowEdit(entropy_regularization=ot_reg_coeff)
                            transport_analysis = analyze_ot_quality(ot_guidance, x0_src, x0_tar)
                            transport_analysis['editing_time'] = editing_time
                            
                            save_enhanced_metrics(transport_analysis, enhanced_save_dir, "transport_analysis.json")
                            
                            if args.verbose and 'transport_cost' in transport_analysis:
                                print(f"📊 Transport cost: {transport_analysis['transport_cost']:.6f}")
                                print(f"📊 Transport entropy: {transport_analysis.get('transport_entropy', 'N/A')}")
                        except Exception as e:
                            print(f"⚠️ Warning: Transport analysis failed: {e}")
                    
                    # Save metadata
                    metadata = {
                        "timestamp": datetime.now().isoformat(),
                        "experiment_name": exp_name,
                        "model_type": model_type,
                        "source_prompt": src_prompt,
                        "target_prompt": tar_prompt,
                        "seed": seed,
                        "editing_time_seconds": editing_time,
                        "parameters": enhanced_params,
                        "enhanced_features": {
                            "optimal_transport": use_optimal_transport,
                            "adaptive_transport": adaptive_transport,
                            "transport_regularization": ot_reg_coeff,
                            "transport_strength": transport_strength
                        },
                        "transport_analysis": transport_analysis
                    }
                    
                    metadata_path = os.path.join(enhanced_save_dir, "metadata.json")
                    with open(metadata_path, "w") as f:
                        json.dump(metadata, f, indent=2)
                    
                    # Run comparison if requested
                    if args.compare_methods:
                        print("🔄 Running comparison with original FlowEdit...")
                        
                        comparison_start = time.time()
                        
                        try:
                            # Run original FlowEdit
                            if model_type == 'SD3':
                                x0_tar_orig = FlowEditSD3(
                                    pipe, scheduler, x0_src, src_prompt, tar_prompt, 
                                    negative_prompt, T_steps, n_avg, src_guidance_scale, 
                                    tar_guidance_scale, n_min, n_max
                                )
                            elif model_type == 'FLUX':
                                x0_tar_orig = FlowEditFLUX(
                                    pipe, scheduler, x0_src, src_prompt, tar_prompt,
                                    negative_prompt, T_steps, n_avg, src_guidance_scale,
                                    tar_guidance_scale, n_min, n_max
                                )
                            
                            # Decode original result
                            x0_tar_orig_denorm = (x0_tar_orig / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
                            with torch.autocast("cuda"), torch.inference_mode():
                                image_tar_orig = pipe.vae.decode(x0_tar_orig_denorm, return_dict=False)[0]
                            image_tar_orig = pipe.image_processor.postprocess(image_tar_orig)
                            
                            comparison_time = time.time() - comparison_start
                            
                            # Save original result
                            original_save_dir = f"{base_save_dir}/original"
                            os.makedirs(original_save_dir, exist_ok=True)
                            
                            original_filename = (f"original_T{T_steps}_avg{n_avg}_"
                                               f"cfg{src_guidance_scale}_{tar_guidance_scale}_"
                                               f"n{n_min}_{n_max}_seed{seed}.png")
                            
                            original_path = os.path.join(original_save_dir, original_filename)
                            image_tar_orig[0].save(original_path)
                            
                            print(f"💾 Original result saved: {original_filename}")
                            print(f"⏱️ Comparison time: Enhanced={editing_time:.2f}s, Original={comparison_time:.2f}s")
                            
                            # Create comparison grid if requested
                            if args.create_comparison_grids:
                                images_dict = {
                                    "Source": image,
                                    "Original FlowEdit": image_tar_orig[0],
                                    "Enhanced FlowEdit": image_tar[0]
                                }
                                
                                comparison_grid_path = os.path.join(base_save_dir, f"comparison_grid_tar{tar_num}_seed{seed}.png")
                                create_comparison_grid(images_dict, comparison_grid_path)
                        
                        except Exception as e:
                            print(f"⚠️ Warning: Comparison with original FlowEdit failed: {e}")
                    
                    print(f"✅ Processing completed for target: {tar_prompt}")
                    
                except Exception as e:
                    print(f"❌ Error during editing: {e}")
                    if args.verbose:
                        import traceback
                        traceback.print_exc()
                    continue

    print(f"\n🎉 All experiments completed successfully!")
    print(f"📁 Results saved in: outputs/{exp_name}/")
    print(f"🔍 Check the enhanced subdirectories for OT-guided results")
    
    # Print summary
    if args.save_transport_analysis:
        print(f"📊 Transport analysis files saved with detailed metrics")
    if args.create_comparison_grids:
        print(f"🖼️ Comparison grids created for visual analysis")
    if args.compare_methods:
        print(f"🔄 Method comparisons completed")

# python run_script_enhanced.py --exp_yaml FLUX_exp_enhanced.yaml --device_number 4
# python run_script_enhanced.py --exp_yaml FLUX_exp_enhanced_bed.yaml --device_number 2

# python run_script_enhanced.py --exp_yaml SD3_exp_enhanced.yaml --device_number 5
# python run_script_enhanced.py --exp_yaml SD3_exp_enhanced_bed.yaml --device_number 3