# Qwen Image Generator

A beautiful, standalone GUI for text-to-image, image editing, and **video generation** using ComfyUI as the backend. Supports multiple models including Qwen-Image and Z-Image Turbo.

## Features

### Image Generation
- **Multiple Models**: Choose between Qwen-Image (Lightning/Quality) or Z-Image Turbo (Fast Photorealistic)
- **Text-to-Image**: Generate images from text prompts
- **Image Editing**: Modify existing images with text descriptions
- **Lightning Mode**: Fast 4-step generation (Qwen)
- **Normal Mode**: High-quality 30-step generation (Qwen)
- **Turbo Mode**: 8-step photorealistic generation (Z-Image)

### Video Generation
- **3 Model Options**: LTX 2B (~30s), Hunyuan 13B (~3min), Wan 14B (~5min)
- **Text-to-Video**: Generate 2.5-7.5 second videos from text descriptions
- **Image-to-Video**: Animate any image with motion (Hunyuan, Wan)
- **Multiple Resolutions**: 480p, 576p, 720p
- **Adjustable Duration**: 41, 81, or 121 frames (~2.5-7.5 seconds)

### User Interface
- Split-view comparison (before/after)
- Real-time generation queue with progress tracking
- Gallery with favorites system
- Quick presets (Portrait, Landscape, Wallpaper, etc.)
- Dark glass-morphism design

### Prompt Tools
- **Local AI Refinement**: Refine, expand, or stylize prompts using Ollama
- **Prompt History**: Search and reuse previous prompts
- **Negative Prompts**: Exclude unwanted elements

### Advanced Options
- Adjustable resolution (512-1536px for images, 480-720p for video)
- Aspect ratio presets (square, portrait, landscape)
- Sampler/scheduler selection
- Seed control for reproducible results
- Batch generation (1-4 images)

## Requirements

- **ComfyUI** running on `http://127.0.0.1:8188`
- **Python 3.10+**
- **Ollama** (optional, for prompt refinement)

### Required Models for Image Generation

Download and place in ComfyUI's `models/` directory:

| Type | Model | Location |
|------|-------|----------|
| Text Encoder | `Qwen2.5-VL-7B-Instruct-abliterated.Q6_K.gguf` | `models/text_encoders/` |
| UNet | `qwen-image-Q6_K.gguf` | `models/unet/` |
| Edit UNet | `qwen-image-edit-2511-Q4_K_M.gguf` | `models/unet/` |
| VAE | `qwen_image_vae.safetensors` | `models/vae/` |
| CLIP Vision | `mmproj-BF16.gguf` | `models/clip_vision/` |
| Lightning LoRA | `Qwen-Image-Lightning-4steps-V1.0.safetensors` | `models/loras/` |

### Required Models for Z-Image Turbo (Optional)

| Type | Model | Location | Size |
|------|-------|----------|------|
| UNet | `z_image_turbo-Q8_0.gguf` | `models/unet/` | 6.7GB |
| Text Encoder | `qwen_3_4b.safetensors` | `models/text_encoders/` | 7.5GB |
| VAE | `ae.safetensors` | `models/vae/` | 320MB |

**Download from:** [Comfy-Org/z_image_turbo](https://huggingface.co/Comfy-Org/z_image_turbo) or [jayn7/Z-Image-Turbo-GGUF](https://huggingface.co/jayn7/Z-Image-Turbo-GGUF)

### Required Models for Video Generation

| Type | Model | Location | Size |
|------|-------|----------|------|
| Video UNet (T2V) | `wan2.1_t2v_14B_bf16.safetensors` | `models/diffusion_models/` | ~27GB |
| Video UNet (I2V) | `wan2.1_i2v_14B_bf16.safetensors` | `models/diffusion_models/` | ~27GB |
| Text Encoder | `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | `models/text_encoders/` | ~4GB |
| Video VAE | `wan_2.1_vae.safetensors` | `models/vae/` | ~1GB |
| CLIP Vision | `clip_vision_h.safetensors` | `models/clip_vision/` | ~2GB |

**Download from:** [Wan-AI/Wan2.1-T2V-14B on Hugging Face](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B)

**Hardware Requirements for Video:**
- Minimum 14GB VRAM (480p, bf16)
- Recommended 24GB+ VRAM for 720p
- ~5 minutes per 5-second video at 480p

## Quick Start

1. Start ComfyUI:
   ```bash
   cd /path/to/ComfyUI
   python main.py
   ```

2. Run the Qwen Image Generator:
   ```bash
   python simple_generator.py
   ```

3. Open `http://localhost:8080` in your browser

Or use the macOS app: double-click `Qwen Image Generator.app`

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Cmd+Enter` | Generate image |
| `Escape` | Cancel generation |

## Video Generation Tips

1. **Start with 480p**: Lower resolutions are much faster to generate
2. **Use short durations first**: 41 frames (~2.5 sec) for testing prompts
3. **Be specific about motion**: Describe camera movements and action
4. **Example prompts**:
   - "A majestic eagle soaring through clouds, aerial tracking shot"
   - "Ocean waves crashing on rocks, slow motion, sunset lighting"
   - "City street at night with rain, neon reflections, cyberpunk"

## Roadmap

- [ ] Custom saveable presets
- [x] Batch generation progress (individual image tracking)
- [x] Video generation support (Wan 2.1)
- [ ] Audio generation support
- [ ] 3D model generation
- [x] Upscaling with LoRAs (2K/4K)
- [x] Multi-angle camera controls (96 positions)
- [ ] Image inpainting

## Changelog

### 2026-01-14 (v2)
- Added **Z-Image Turbo** model support (6B, fast photorealistic)
- Model selector UI in Generate tab (Qwen vs Z-Image)
- Added 1536px resolution option
- Three video models: LTX 2B, Hunyuan 13B, Wan 14B

### 2026-01-14
- Added **Video Generation** tab with Wan 2.1 support
- Text-to-Video (T2V) and Image-to-Video (I2V) modes
- Multiple resolution options (480p, 576p, 720p)
- Adjustable video duration (2.5-7.5 seconds)
- Video prompt refinement with Ollama

### 2026-01-13 (v2)
- Upgraded image gen model: Q4_K_M → Q6_K
- Removed all cloud functionality
- Added gallery filters (Lightning/Normal/Edit)
- Added compare mode (2 images side-by-side)
- Increased edit timeout to 10 minutes

## License

MIT License - See ComfyUI for underlying framework license.
