<div align="center">

# Optimal Transport for Rectified Flow Image Editing

**Unifying Inversion-Based and Direct Methods**

<a href='https://arxiv.org/abs/2508.02363'><img src='https://img.shields.io/badge/arXiv-Preprint-red'></a>
<a href='https://wacv.thecvf.com/virtual/2026/poster/955'><img src='https://img.shields.io/badge/WACV-2026-blue'></a>
<a href='https://www.youtube.com/watch?v=ceJxfEnBIxQ'><img src='https://img.shields.io/badge/YouTube-Video-red?logo=youtube'></a>
<a href='https://wacv.thecvf.com/media/wacv-2026/Slides/955_1nB2xpY.pdf'><img src='https://img.shields.io/badge/Slides-PDF-orange'></a>
<a href='https://huggingface.co/papers/2508.02363'><img src='https://img.shields.io/badge/%F0%9F%A4%97-Paper%20Page-yellow'></a>
<a href='https://huggingface.co/maryanlup/OT-RF'><img src='https://img.shields.io/badge/%F0%9F%A4%97-Model-yellow'></a>
<a href='https://huggingface.co/spaces/maryanlup/OT-RF'><img src='https://img.shields.io/badge/%F0%9F%A4%97-Demo-yellow'></a>
![GitHub stars](https://img.shields.io/github/stars/marianlupascu/OT-RF?style=social)
<img src="https://hits.sh/github.com/marianlupascu/OT-RF.svg?label=visitors&color=527f8d" alt="Visitors">

**WACV 2026 - Algorithms Track**

</div>

---

We introduce **Transport-Guided Rectified Flow Image Editing**, a unified framework for semantic image editing that combines rectified flows with optimal transport theory. Our approach includes both inversion-based editing (RF-Inversion+OTC) and inversion-free editing (FlowEdit+OTC) methods. <p  align="center"> <img  src="assets/teaser.png"  alt="teaser"  width="80%"> </p>

## 🔥 Updates

* **2026-11-17**: Released on 🤗 Hugging Face: [model](https://huggingface.co/maryanlup/OT-RF) · [demo](https://huggingface.co/spaces/maryanlup/OT-RF)
* **2026-03-10**: Presented at WACV 2026 in Tucson, Arizona. See the [official poster](https://wacv.thecvf.com/virtual/2026/poster/955)
* **2025-11-06**: Paper accepted to WACV 2026
* **2025-07-11**: Paper submitted to WACV 2026
* **2025-07-08**: Initial release of code + demo
* **2025-07-01**: Paper available on [arXiv paper](https://arxiv.org/abs/2508.02363)


## 🚀 Quick Start  

### Transport-Guided RF Inversion 
We extend [🤗 Hugging Face Diffusers](https://github.com/huggingface/diffusers) to support **OT-guided rectified flow inversion and editing** with FLUX models.

```python
import os, sys, torch
from diffusers import FluxPipeline
from diffusers.utils import load_image
from huggingface_hub import hf_hub_download

torch.manual_seed(42)

base_pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16,
    custom_pipeline="pipeline_flux_rf_inversion").to("cuda")
sys.modules["pipeline_flux_rf_inversion"] = sys.modules[type(base_pipe).__module__]

sys.path.append(os.path.dirname(hf_hub_download("maryanlup/OT-RF", "improved_optimal_transport_rf_inversion.py")))
from improved_optimal_transport_rf_inversion import OptimalTransportRFInversionPipeline
ot_pipe = OptimalTransportRFInversionPipeline.from_pipe(base_pipe)

image = load_image("https://raw.githubusercontent.com/marianlupascu/OT-RF/main/assets/face_input.jpg").resize((1024, 1024))
inverted_latents, image_latents, latent_image_ids = ot_pipe.invert(image=image, num_inversion_steps=28)

edited_image = ot_pipe(
    prompt="a person wearing glasses",
    inverted_latents=inverted_latents, image_latents=image_latents, latent_image_ids=latent_image_ids,
    start_timestep=0, stop_timestep=7/28, num_inference_steps=28, eta=0.9,
    use_ot_guidance=True, transport_strength=0.3, ot_guidance_phase=0.3,
    respect_prompt_guidance=False,
).images[0]
edited_image.save("edited_ot.png")
```
----------

### Transport-Enhanced FlowEdit (Inversion-Free)

#### 1. Setup Configuration Files

**SD3 configuration (`SD3_exp_enhanced.yaml`):**
```
-
  exp_name: "demo_sd3_enhanced"
  model_type: "SD3"
  T_steps: 50
  n_avg: 1
  src_guidance_scale: 3.5
  tar_guidance_scale: 13.5
  n_min: 0
  n_max: 33
  seed: 42

  use_optimal_transport: true
  transport_strength: 0.6            
  entropy_regularization: 0.01       
  use_regional_transport: false
  
  dataset_yaml: "edits.yaml"
```

**FLUX configuration (`FLUX_exp_enhanced.yaml`):**
```
-
  exp_name: "demo_flux_enhanced"
  model_type: "FLUX"
  T_steps: 50
  n_avg: 1
  src_guidance_scale: 3.5
  tar_guidance_scale: 13.5
  n_min: 0
  n_max: 33
  seed: 42

  use_optimal_transport: true
  transport_strength: 0.6           
  entropy_regularization: 0.01      
  use_regional_transport: false
  
  dataset_yaml: "edits.yaml"
```

**Dataset configuration (`edits.yaml`):**
```
-
    input_img:  SFHQ_pt1_00000008.jpg
    source_prompt: face of a man/woman

    target_prompts:
    - face of a man/woman wearing glasses
    
- 
    input_img:  SFHQ_pt1_00000009.jpg
    source_prompt: face of a man/woman

    target_prompts:
    - face of a man/woman wearing glasses

```
----------

#### 2. Run Transport-Enhanced FlowEdit

**Basic usage:**

```# Run with SD3 
python run_script_enhanced.py --exp_yaml SD3_exp_enhanced.yaml --device_number 0 

# Run with FLUX 
python run_script_enhanced.py --exp_yaml FLUX_exp_enhanced.yaml --device_number 0
```

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

📄 More details in the [arXiv paper](https://arxiv.org/abs/2508.02363)

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
@InProceedings{Lupascu_2026_WACV,
    author    = {Lupa\c{s}cu, Marian and Stupariu, Mihai Sorin},
    title     = {Optimal Transport for Rectified Flow Image Editing: Unifying Inversion-Based and Direct Methods},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {March},
    year      = {2026},
    pages     = {6764--6774},
    doi       = {10.1109/WACV61042.2026.00654}
}
```

---

## 🙏 Acknowledgments

* Built upon [FLUX](https://github.com/black-forest-labs/flux) rectified flow models
* Extends [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
* Inspired by optimal transport theory, [FlowEdit](https://github.com/fallenshock/FlowEdit) and [RF-Inversion](https://rf-inversion.github.io/)

---
