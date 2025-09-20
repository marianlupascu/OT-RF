<div align="center">

# Transport-Guided Rectified Flow

**Improved Image Editing Using Optimal Transport Theory**

<a href='TODO'><img src='https://img.shields.io/badge/Project-Page-green'></a> <a href='TODO'>
<img src='https://img.shields.io/badge/WACV-2026-blue'></a> 
<a href='TODO'><img src='https://img.shields.io/badge/arXiv-Preprint-red'></a>
![GitHub stars](https://img.shields.io/github/stars/marianlupascu/OT-Inversion?style=social)

**WACV 2026 Submission - Algorithms Track**

</div>

---

We introduce **Transport-Guided Rectified Flow Image Editing**, a unified framework for semantic image editing that combines rectified flows with optimal transport theory. Our approach includes both inversion-based editing (RF-Inversion+OTC) and inversion-free editing (FlowEdit+OTC) methods. <p  align="center"> <img  src="assets/teaser.png"  alt="teaser"  width="80%"> </p>

## 🔥 Updates

* **2025-07-11**: Paper submitted to WACV 2026
* **2025-07-08**: Initial release of code + demo
* **2025-07-01**: Paper available on [arXiv](TODO)


## 🚀 Quick Start  

### Transport-Guided RF Inversion 
We extend [🤗 Hugging Face Diffusers](https://github.com/huggingface/diffusers) to support **OT-guided rectified flow inversion and editing** with FLUX models.

```
import torch, random, numpy as np, requests, os
from diffusers import FluxPipeline
from io import BytesIO
from PIL import Image
from optimal_transport_rf_inversion import OptimalTransportRFInversionPipeline

# Reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Load base FLUX model
base_pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
    custom_pipeline="pipeline_flux_rf_inversion"
)
base_pipe.to("cuda")

# Wrap with OT-guided inversion
ot_pipe = OptimalTransportRFInversionPipeline.from_pipe(base_pipe)

# Load image
def download_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")

img_url = "https://example.com/sample_image.jpg"
original_image = download_image(img_url).resize((512, 512))

# Invert with optimal transport guidance
inverted_latents, image_latents, latent_image_ids = ot_pipe.invert(
    image=original_image,
    num_inversion_steps=28,
    transport_strength=0.5,
    adaptive_scheduling=True
)

# Edit with transport-guided flow
edited_image = ot_pipe(
    prompt="a sleeping cat",
    inverted_latents=inverted_latents,
    image_latents=image_latents,
    latent_image_ids=latent_image_ids,
    start_timestep=0,
    stop_timestep=7/28,
    num_inference_steps=28,
    eta=0.9,
    use_ot_guidance=True,
    transport_strength=0.3,
    ot_guidance_phase=0.3,
    respect_prompt_guidance=False
).images[0]

# Save results
os.makedirs("./results", exist_ok=True)
edited_image.save("./results/edited_ot.png")
print("✅ Saved to ./results/edited_ot.png")` 
```
----------

### Transport-Enhanced FlowEdit (Inversion-Free)

#### 1. Setup Configuration Files

**SD3 configuration (`SD3_exp_enhanced.yaml`):**
```
`exp_name:  "demo_sd3_enhanced"  model_type:  "SD3"  T_steps:  50  n_avg:  1  src_guidance_scale:  3.5  tar_guidance_scale:  13.5  n_min:  0  n_max:  33  seed:  42  use_optimal_transport:  true  transport_strength:  0.6  ot_reg_coeff:  0.01  adaptive_transport:  true  dataset_yaml:  "edits.yaml"` 
```
**FLUX configuration (`FLUX_exp_enhanced.yaml`):**
```
`exp_name:  "demo_flux_enhanced"  model_type:  "FLUX"  T_steps:  28  n_avg:  1  src_guidance_scale:  1.5  tar_guidance_scale:  7.5  n_min:  0  n_max:  24  seed:  42  use_optimal_transport:  true  transport_strength:  1.0  ot_reg_coeff:  0.1  adaptive_transport:  true  dataset_yaml:  "edits.yaml"` 
```
**Dataset configuration (`edits.yaml`):**
```
`-  input_img:  SFHQ_pt1_00000008.jpg  source_prompt:  face  of  a  man/woman  target_prompts:  -  face  of  a  man/woman  wearing  glasses  -  input_img:  SFHQ_pt1_00000009.jpg  source_prompt:  face  of  a  man/woman  target_prompts:  -  face  of  a  man/woman  wearing  glasses` 
```
----------

#### 2. Run Transport-Enhanced FlowEdit

**Basic usage:**

`# Run with SD3 python run_script_enhanced.py --exp_yaml SD3_exp_enhanced.yaml --device_number 0 # Run with FLUX python run_script_enhanced.py --exp_yaml FLUX_exp_enhanced.yaml --device_number 0` 

**Advanced usage with analysis:**

`python run_script_enhanced.py \
    --exp_yaml FLUX_exp_enhanced.yaml \
    --device_number 0 \
    --save_transport_analysis \
    --compare_methods \
    --create_comparison_grids \
    --verbose` 

#### 3. Compare with Original FlowEdit

`# Original FlowEdit (no OT) python run_script.py --exp_yaml FLUX_exp.yaml --device_number 0` 

----------

### 4. Key Parameters

**Transport Enhancement:**

-   `use_optimal_transport`: Enable/disable OT guidance (`true/false`)
    
-   `transport_strength`: Base transport strength (0.1–1.2)
    
-   `ot_reg_coeff`: Regularization for Sinkhorn solver (0.01–0.15)
    
-   `adaptive_transport`: Enable adaptive transport strength (`true/false`)
    

**Architecture-Specific Optimal Settings:**

-   FLUX: `transport_strength=1.0`, `ot_reg_coeff=0.1`
    
-   SD3: `transport_strength=0.4`, `ot_reg_coeff=0.15`
    

---

## 📊 Results

| Task                      | Metric     | RF-Inversion | **Ours (OT)** | Improvement |
| ------------------------- | ---------- | ------------ | ------------- | ----------- |
| Stroke-to-Image (Bedroom) | L2 ↓       | 82.55        | **76.10**     | 7.8%        |
| Stroke-to-Image (Church)  | L2 ↓       | 80.35        | **69.97**     | 12.9%       |
| Face Editing              | Identity ↓ | 0.387        | **0.112**     | 71.1%       |
| Face Editing              | CLIP-I ↑   | 0.936        | **0.999**     | 6.7%        |
| Reconstruction            | SSIM ↑     | 0.833        | **0.992**     | 19.1%       |

📄 More details in the [arXiv paper](TODO)

---

## 📦 Requirements

* Python ≥ 3.9
* PyTorch ≥ 2.0
* CUDA ≥ 11.8
* Diffusers ≥ 0.24.0
* Transformers ≥ 4.36.0

---

## 📖 Citation

```bibtex
@inproceedings{otrf_2025,
  title     = {Optimal Transport for Rectified Flow Image Editing: Unifying Inversion-Based and Direct Methods},
  author    = {Anonymous},
}
```

---

## 📄 License

This research is submitted to WACV 2026. Code will be released under MIT License upon acceptance.

---

## 🙏 Acknowledgments

* Built upon [FLUX](https://github.com/black-forest-labs/flux) rectified flow models
* Extends [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
* Inspired by optimal transport theory and [RF-Inversion](https://rf-inversion.github.io/)

---