# Qwen Image Generator

A beautiful, standalone GUI for Qwen-Image-2512 text-to-image and image editing using ComfyUI as the backend.

## Features

### Image Generation
- **Text-to-Image**: Generate images from text prompts
- **Image Editing**: Modify existing images with text descriptions
- **Lightning Mode**: Fast 4-step generation for quick previews
- **Normal Mode**: High-quality 20-step generation

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
- Adjustable resolution (512-1024px)
- Aspect ratio presets (square, portrait, landscape)
- Sampler/scheduler selection
- Seed control for reproducible results
- Batch generation (1-4 images)

## Requirements

- **ComfyUI** running on `http://127.0.0.1:8188`
- **Python 3.10+**
- **Ollama** (optional, for prompt refinement)

### Required Models

Download and place in ComfyUI's `models/` directory:

| Type | Model | Location |
|------|-------|----------|
| Text Encoder | `Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf` | `models/text_encoders/` |
| UNet | `qwen-image-2512-Q4_K_M.gguf` | `models/unet/` |
| Edit UNet | `qwen-image-edit-2511-Q4_K_M.gguf` | `models/unet/` |
| VAE | `qwen_image_vae.safetensors` | `models/vae/` |
| CLIP Vision | `mmproj-BF16.gguf` | `models/clip_vision/` |
| Lightning LoRA | `Qwen-Image-Lightning-4steps-V1.0.safetensors` | `models/loras/` |

See `requirements-qwen.txt` for download links.

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

## Roadmap

- [ ] Custom saveable presets
- [x] Batch generation progress (individual image tracking)
- [ ] Video generation support
- [ ] Audio generation support
- [ ] 3D model generation
- [x] Upscaling with LoRAs (2K/4K)
- [x] Multi-angle camera controls (96 positions)
- [ ] Image inpainting

## License

MIT License - See ComfyUI for underlying framework license.
