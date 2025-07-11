<div align="center">

# Transport-Guided Rectified Flow Inversion

**Improved Image Editing Using Optimal Transport Theory**

<a href='TODO'><img src='https://img.shields.io/badge/Project-Page-green'></a> <a href='TODO'>
<img src='https://img.shields.io/badge/WACV-2026-blue'></a> 
<a href='TODO'><img src='https://img.shields.io/badge/arXiv-Preprint-red'></a>
![GitHub stars](https://img.shields.io/github/stars/marianlupascu/OT-Inversion?style=social)

**WACV 2026 Submission - Algorithms Track**

</div>

---

We introduce **Transport-Guided Rectified Flow Inversion**, a zero-shot method for semantic image editing that combines rectified flows with optimal transport theory. Unlike existing RF inversion approaches, our method avoids optimization or fine-tuning and provides controllable, high-fidelity edits while preserving structure.

<p align="center">
  <img src="assets/teaser.png" alt="teaser" width="80%">
</p>

---

## 🔥 Updates

* **2025-07-11**: Paper submitted to WACV 2026
* **2025-07-08**: Initial release of code + demo
* **2025-07-01**: Paper available on [arXiv](TODO)

---

## 🚀 Diffusers Implementation

We extend [🤗 Hugging Face Diffusers](https://github.com/huggingface/diffusers) to support **OT-guided rectified flow inversion and editing** with FLUX models.

### ✨ Quick Example

```python
import torch, random, numpy as np, requests, os
from diffusers import FluxPipeline
from io import BytesIO
from PIL import Image
from optimal_transport_rf_inversion import OptimalTransportRFInversionPipeline

# Reproducibility
random.seed(42); np.random.seed(42)
torch.manual_seed(42); torch.cuda.manual_seed(42)

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
print("✅ Saved to ./results/edited_ot.png")
```

### 🎯 Advanced Usage

```python
# Stroke-to-image
stroke_image = Image.open("./assets/stroke_bedroom.png")
realistic_image = ot_pipe.stroke_to_image(
    stroke_image=stroke_image,
    prompt="a photo-realistic picture of a bedroom",
    transport_strength=0.4,
    num_steps=28
)

# Face editing
face_image = Image.open("./assets/face_input.jpg")
edited_face = ot_pipe.edit_face(
    image=face_image,
    prompt="a smiling cartoon",
    preserve_identity=True,
    transport_strength=0.2
)
```

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

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

---

## 📖 Citation

```bibtex
@inproceedings{transport_rf_inversion_2026,
  title     = {Transport-Guided Rectified Flow Inversion: Improved Image Editing Using Optimal Transport Theory},
  author    = {Anonymous},
  booktitle = {IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2026},
  organization = {IEEE}
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
