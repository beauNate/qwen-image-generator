#!/usr/bin/env python3
"""
Qwen Image Generator - Enhanced Edition
Run this and open http://localhost:8080
Access from other devices: http://<your-ip>:8080
"""

import json
import urllib.request
import urllib.parse
import os
import sys
import time
import threading
import webbrowser
import base64
import uuid
from http.server import HTTPServer, SimpleHTTPRequestHandler
import subprocess

COMFYUI_URL = "http://127.0.0.1:8188"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
FAVORITES_FILE = os.path.join(os.path.dirname(__file__), "favorites.json")
HISTORY_FILE = os.path.join(os.path.dirname(__file__), "prompt_history.json")

# Note: Seed storage moved to client-side localStorage

# Prompt history (in-memory, synced to file)
prompt_history = []

# Quick presets
PRESETS = {
    'quick': {'mode': 'lightning', 'resolution': 512, 'aspect': 'square', 'name': 'Quick Preview'},
    'portrait': {'mode': 'lightning', 'resolution': 768, 'aspect': 'portrait', 'name': 'Portrait'},
    'landscape': {'mode': 'lightning', 'resolution': 768, 'aspect': 'landscape', 'name': 'Landscape'},
    'wallpaper': {'mode': 'lightning', 'resolution': 1024, 'aspect': 'landscape', 'name': 'Wallpaper'},
    'quality': {'mode': 'normal', 'resolution': 768, 'aspect': 'square', 'name': 'High Quality'},
    'hd_quality': {'mode': 'normal', 'resolution': 1024, 'aspect': 'square', 'name': 'HD Quality'},
}

# Sampler options for advanced users
SAMPLERS = ['euler', 'euler_ancestral', 'heun', 'dpm_2', 'dpm_2_ancestral', 'lms', 'dpm_fast', 'dpm_adaptive', 'dpmpp_2s_ancestral', 'dpmpp_sde', 'dpmpp_2m']
SCHEDULERS = ['normal', 'karras', 'exponential', 'sgm_uniform', 'simple', 'ddim_uniform']

# Workflow templates
def get_workflow(mode='lightning', resolution=512, aspect='square', seed=None, negative_prompt='', sampler='euler', scheduler='normal'):
    """Generate workflow based on settings"""

    # Calculate dimensions based on aspect ratio
    if aspect == 'landscape':
        width = resolution
        height = int(resolution * 0.66)
    elif aspect == 'portrait':
        width = int(resolution * 0.66)
        height = resolution
    else:  # square
        width = resolution
        height = resolution

    # Use provided seed or generate random one
    if seed is None:
        seed = int(time.time() * 1000) % 999999999

    steps = 4 if mode == 'lightning' else 30
    cfg = 1.0 if mode == 'lightning' else 5.0

    workflow = {
        "3": {
            "class_type": "CLIPLoaderGGUF",
            "inputs": {
                "clip_name": "Qwen2.5-VL-7B-Instruct-abliterated.Q6_K.gguf",
                "type": "qwen_image"
            }
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "PROMPT_PLACEHOLDER",
                "clip": ["3", 0]
            }
        },
        "5": {
            "class_type": "UnetLoaderGGUF",
            "inputs": {
                "unet_name": "qwen-image-Q6_K.gguf"
            }
        },
        "6": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": "qwen_image_vae.safetensors"
            }
        },
        "7": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1
            }
        },
        "9": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": negative_prompt,
                "clip": ["3", 0]
            }
        },
        "10": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["8", 0],
                "vae": ["6", 0]
            }
        },
        "11": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "qwen_" + mode,
                "images": ["10", 0]
            }
        }
    }

    # Add LoRA for lightning mode
    if mode == 'lightning':
        workflow["12"] = {
            "class_type": "LoraLoader",
            "inputs": {
                "lora_name": "Qwen-Image-Lightning-4steps-V1.0.safetensors",
                "strength_model": 1.0,
                "strength_clip": 1.0,
                "model": ["5", 0],
                "clip": ["3", 0]
            }
        }
        workflow["8"] = {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler,
                "scheduler": scheduler,
                "denoise": 1.0,
                "model": ["12", 0],
                "positive": ["4", 0],
                "negative": ["9", 0],
                "latent_image": ["7", 0]
            }
        }
    else:
        workflow["8"] = {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler,
                "scheduler": scheduler,
                "denoise": 1.0,
                "model": ["5", 0],
                "positive": ["4", 0],
                "negative": ["9", 0],
                "latent_image": ["7", 0]
            }
        }

    return workflow, seed


def get_zimage_workflow(resolution=1024, aspect='square', seed=None, negative_prompt=''):
    """Generate workflow for Z-Image Turbo (6B parameter model, fast photorealistic generation)"""

    # Calculate dimensions based on aspect ratio (must be divisible by 32)
    if aspect == 'landscape':
        width = resolution
        height = int(resolution * 0.66)
        # Round to nearest 32
        height = (height // 32) * 32
    elif aspect == 'portrait':
        width = int(resolution * 0.66)
        height = resolution
        # Round to nearest 32
        width = (width // 32) * 32
    else:  # square
        width = resolution
        height = resolution

    # Use provided seed or generate random one
    if seed is None:
        seed = int(time.time() * 1000) % 999999999

    # Z-Image Turbo settings: 8 steps, CFG 1.0 (turbo doesn't need guidance)
    steps = 8
    cfg = 1.0

    workflow = {
        # Text Encoder - Qwen 3 4B (type: lumina2)
        "1": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "qwen_3_4b.safetensors",
                "type": "lumina2"
            }
        },
        # Positive prompt encoding
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "PROMPT_PLACEHOLDER",
                "clip": ["1", 0]
            }
        },
        # Negative prompt encoding (Z-Image uses ConditioningZeroOut for negative)
        "3": {
            "class_type": "ConditioningZeroOut",
            "inputs": {
                "conditioning": ["2", 0]
            }
        },
        # Load Z-Image Turbo UNet (GGUF quantized)
        "4": {
            "class_type": "UnetLoaderGGUF",
            "inputs": {
                "unet_name": "z_image_turbo-Q8_0.gguf"
            }
        },
        # ModelSamplingAuraFlow for Z-Image
        "5": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {
                "shift": 1.73,
                "model": ["4", 0]
            }
        },
        # Load Flux VAE (ae.safetensors)
        "6": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": "ae.safetensors"
            }
        },
        # Empty SD3 latent (Z-Image uses SD3 latent format)
        "7": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1
            }
        },
        # KSampler
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "sgm_uniform",
                "denoise": 1.0,
                "model": ["5", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["7", 0]
            }
        },
        # VAE Decode
        "9": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["8", 0],
                "vae": ["6", 0]
            }
        },
        # Save Image
        "10": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "zimage_turbo",
                "images": ["9", 0]
            }
        }
    }

    return workflow, seed


HTML_PAGE = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Qwen Image Generator</title>
    <style>
        /* ============================================
           DESIGN SYSTEM: Apple Glassmorphism - Refined
           - Consistent 8px base spacing grid
           - Unified font sizing scale
           - Smooth animations on ALL interactive elements
           - Frosted glass with proper depth
           ============================================ */

        :root {
            /* Spacing - Strict 8px grid (HIG compliant) */
            --space-1: 4px;
            --space-2: 8px;
            --space-3: 8px;   /* Alias to space-2 for compatibility */
            --space-4: 16px;
            --space-5: 16px;  /* Alias to space-4 for compatibility */
            --space-6: 24px;
            --space-8: 32px;

            /* Glass Colors */
            --glass-bg: rgba(255, 255, 255, 0.07);
            --glass-bg-hover: rgba(255, 255, 255, 0.12);
            --glass-border: rgba(255, 255, 255, 0.12);
            --glass-border-hover: rgba(255, 255, 255, 0.2);

            /* Text - 4-level hierarchy */
            --text-primary: rgba(255, 255, 255, 0.95);
            --text-secondary: rgba(255, 255, 255, 0.7);
            --text-tertiary: rgba(255, 255, 255, 0.5);
            --text-quaternary: rgba(255, 255, 255, 0.3);

            /* Accent - Apple Blue */
            --accent: #0A84FF;
            --accent-hover: #409CFF;
            --accent-glow: rgba(10, 132, 255, 0.35);
            --accent-bg: rgba(10, 132, 255, 0.12);
            --accent-bg-hover: rgba(10, 132, 255, 0.15);
            --accent-bg-active: rgba(10, 132, 255, 0.25);
            --accent-bg-subtle: rgba(10, 132, 255, 0.1);
            --accent-bg-medium: rgba(10, 132, 255, 0.2);
            --accent-border: rgba(10, 132, 255, 0.25);
            --accent-border-hover: rgba(10, 132, 255, 0.4);
            --accent-ring: rgba(10, 132, 255, 0.2);

            /* Semantic */
            --success: #30D158;
            --success-bg: rgba(48, 209, 88, 0.12);
            --success-glow: rgba(48, 209, 88, 0.35);
            --warning: #FFD60A;
            --error: #FF453A;
            --error-bg: rgba(255, 69, 58, 0.12);
            --error-bg-hover: rgba(255, 69, 58, 0.15);
            --error-bg-active: rgba(255, 69, 58, 0.25);
            --error-border: rgba(255, 69, 58, 0.3);
            --error-border-hover: rgba(255, 69, 58, 0.5);
            --error-muted: rgba(255, 69, 58, 0.8);
            --purple: #A855F7;  /* Apple systemPurple-like */

            /* Typography - Apple HIG scale (strict) */
            --font-sans: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", system-ui, sans-serif;
            --font-mono: "SF Mono", "Fira Code", ui-monospace, monospace;
            --text-caption: 12px;   /* Caption 2 */
            --text-footnote: 13px;  /* Footnote */
            --text-subhead: 15px;   /* Subhead */
            --text-body: 17px;      /* Body - THE standard */
            --text-title3: 20px;    /* Title 3 */
            --text-title2: 22px;    /* Title 2 */
            --text-title1: 28px;    /* Title 1 */
            --text-largetitle: 34px;/* Large Title */

            /* Radius - Apple's signature */
            --radius-sm: 8px;
            --radius-md: 12px;
            --radius-lg: 16px;
            --radius-xl: 20px;
            --radius-full: 9999px;

            /* Animation - Unified timing */
            --ease-out: cubic-bezier(0.25, 1, 0.5, 1);
            --ease-spring: cubic-bezier(0.25, 1, 0.5, 1);  /* Same as ease-out - no bounce in pro tools */
            --duration-fast: 150ms;
            --duration-base: 200ms;
            --duration-slow: 300ms;
        }

        /* Reset with smooth defaults */
        *, *::before, *::after {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        /* Smooth scrolling */
        html {
            scroll-behavior: smooth;
        }

        body {
            font-family: var(--font-sans);
            font-size: var(--text-body);  /* 17px - Apple HIG body standard */
            line-height: 1.4;             /* Apple standard */
            color: var(--text-primary);
            max-width: 600px;
            margin: 0 auto;
            padding: var(--space-4);
            background: #1c1c1e;          /* Apple systemBackground (dark) - solid, no gradients */
            min-height: 100vh;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }

        /* Responsive - mobile first */
        @media (max-width: 640px) {
            body {
                padding: var(--space-2);
                max-width: 100%;
            }
        }

        /* Page load animation */
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(16px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        body {
            animation: fadeIn 0.4s var(--ease-out);
        }

        h1 {
            text-align: center;
            margin-bottom: var(--space-1);
            font-size: var(--text-title1);
            font-weight: 700;
            letter-spacing: -0.02em;
            background: linear-gradient(135deg, #fff 0%, rgba(255,255,255,0.8) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: fadeInUp 0.4s var(--ease-out) 0.1s both;
        }

        .subtitle {
            text-align: center;
            color: var(--text-tertiary);
            margin-bottom: var(--space-6);
            font-size: var(--text-subhead);
            font-weight: 400;
            animation: fadeInUp 0.4s var(--ease-out) 0.15s both;
        }

        /* Connection Status */
        .connection-status {
            display: inline-flex;
            align-items: center;
            gap: var(--space-2);
            font-size: var(--text-caption);
            color: var(--text-quaternary);
            margin-left: var(--space-2);
        }
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--text-quaternary);
            transition: background var(--duration-slow);
        }
        .status-dot.connected { background: var(--success); box-shadow: 0 0 6px var(--success); }
        .status-dot.disconnected { background: var(--error); box-shadow: 0 0 6px var(--error); }
        .status-dot.checking { background: var(--warning); animation: pulse 1s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }

        /* Toast Notifications */
        .toast-container {
            position: fixed;
            top: var(--space-5);
            right: var(--space-5);
            z-index: 10000;
            display: flex;
            flex-direction: column;
            gap: var(--space-2);
            pointer-events: none;
        }
        .toast {
            background: var(--glass-bg);
            backdrop-filter: blur(20px) saturate(180%);
            -webkit-backdrop-filter: blur(20px) saturate(180%);
            border: 1px solid var(--glass-border);
            border-radius: var(--radius-md);
            padding: var(--space-3) var(--space-4);
            display: flex;
            align-items: center;
            gap: var(--space-2);
            min-width: 250px;
            max-width: 400px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            animation: toastIn 0.3s var(--ease-out);
            pointer-events: auto;
        }
        .toast.hiding {
            animation: toastOut 0.3s var(--ease-out) forwards;
        }
        .toast-icon {
            font-size: var(--text-body);
            flex-shrink: 0;
        }
        .toast-content {
            flex: 1;
        }
        .toast-title {
            font-weight: 600;
            font-size: var(--text-footnote);
            color: var(--text-primary);
        }
        .toast-message {
            font-size: var(--text-caption);
            color: var(--text-secondary);
            margin-top: var(--space-1);
        }
        .toast.success { border-left: 3px solid var(--success); }
        .toast.error { border-left: 3px solid var(--error); }
        .toast.warning { border-left: 3px solid var(--warning); }
        .toast.info { border-left: 3px solid var(--accent); }
        @keyframes toastIn {
            from { opacity: 0; transform: translateX(24px); }
            to { opacity: 1; transform: translateX(0); }
        }
        @keyframes toastOut {
            from { opacity: 1; transform: translateX(0); }
            to { opacity: 0; transform: translateX(24px); }
        }

        /* Glass Card Base */
        .glass {
            background: var(--glass-bg);
            backdrop-filter: blur(20px) saturate(180%);
            -webkit-backdrop-filter: blur(20px) saturate(180%);
            border: 1px solid var(--glass-border);
            border-radius: var(--radius-lg);
        }

        /* Tabs - Pill style like iOS */
        .tabs {
            display: flex;
            margin-bottom: var(--space-6);
            background: var(--glass-bg);
            backdrop-filter: blur(20px) saturate(180%);
            -webkit-backdrop-filter: blur(20px) saturate(180%);
            border: 1px solid var(--glass-border);
            border-radius: var(--radius-xl);
            padding: var(--space-1);
            gap: var(--space-1);
            animation: fadeInUp 0.4s var(--ease-out) 0.2s both;
        }

        .tab {
            flex: 1;
            padding: var(--space-2) var(--space-4);
            text-align: center;
            background: transparent;
            cursor: pointer;
            border: none;
            color: var(--text-tertiary);
            font-size: var(--text-subhead);
            font-weight: 500;
            border-radius: var(--radius-lg);
            transition: all var(--duration-base) var(--ease-out);
        }

        .tab:hover {
            color: var(--text-secondary);
            background: rgba(255, 255, 255, 0.05);
        }

        .tab.active {
            background: rgba(255, 255, 255, 0.15);
            color: var(--text-primary);
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        }

        .tab-content { display: none; }
        .tab-content.active {
            display: block;
            animation: fadeIn 0.25s var(--ease-out);
        }

        /* Input sections - Glass Card */
        .input-section {
            background: var(--glass-bg);
            backdrop-filter: blur(20px) saturate(180%);
            -webkit-backdrop-filter: blur(20px) saturate(180%);
            border: 1px solid var(--glass-border);
            border-radius: var(--radius-xl);
            padding: var(--space-6);
            margin-bottom: var(--space-4);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
            animation: fadeInUp 0.4s var(--ease-out) 0.25s both;
        }

        .input-section > h2:first-child {
            margin-top: 0;
            margin-bottom: var(--space-4);
        }

        label {
            display: block;
            margin-bottom: var(--space-2);
            font-weight: 500;
            font-size: var(--text-footnote);
            color: var(--text-secondary);
            letter-spacing: 0.01em;
        }

        textarea, input[type="text"], input[type="number"] {
            width: 100%;
            padding: var(--space-3) var(--space-4);
            border: 1px solid var(--glass-border);
            border-radius: var(--radius-md);
            font-size: var(--text-subhead);
            font-family: var(--font-sans);
            background: rgba(0, 0, 0, 0.2);
            color: var(--text-primary);
            margin-bottom: var(--space-3);
            transition: all var(--duration-base) var(--ease-out);
        }

        textarea { height: 112px; resize: vertical; overflow-y: auto; }
        textarea.textarea-short { height: 64px; }

        textarea:focus, input:focus, select:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-glow);
            background: rgba(0, 0, 0, 0.25);
        }

        textarea::placeholder, input::placeholder {
            color: var(--text-quaternary);
        }

        /* Options */
        .options-row {
            display: flex;
            gap: var(--space-3);
            margin-bottom: var(--space-3);
            flex-wrap: wrap;
        }
        .options-row-spaced {
            display: flex;
            gap: var(--space-3);
            margin-top: var(--space-3);
            margin-bottom: var(--space-3);
            flex-wrap: wrap;
        }
        .option-group {
            flex: 1;
            min-width: 130px;
            display: flex;
            flex-direction: column;
        }
        .option-group label {
            display: block;
            margin-bottom: var(--space-1);
            font-size: var(--text-footnote);
            color: var(--text-secondary);
            font-weight: 500;
        }
        .option-group .option-hint {
            display: block;
            margin-bottom: var(--space-2);
            font-size: var(--text-caption);
            color: var(--text-tertiary);
        }
        .option-group select,
        .option-group input {
            flex: 1;
        }

        select {
            width: 100%;
            padding: var(--space-2) var(--space-3);
            border: 1px solid var(--glass-border);
            border-radius: var(--radius-md);
            font-size: var(--text-subhead);
            font-family: var(--font-sans);
            background: rgba(0, 0, 0, 0.2);
            color: var(--text-primary);
            cursor: pointer;
            transition: all var(--duration-base) var(--ease-out);
            -webkit-appearance: none;
            appearance: none;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' fill='rgba(255,255,255,0.5)' viewBox='0 0 16 16'%3E%3Cpath d='M8 11L3 6h10l-5 5z'/%3E%3C/svg%3E");
            background-repeat: no-repeat;
            background-position: right 12px center;
            padding-right: 36px;
        }

        select:hover { border-color: var(--glass-border-hover); }
        select:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-glow);
        }

        .time-estimate {
            font-size: var(--text-footnote);
            color: var(--text-secondary);
            margin-bottom: var(--space-3);
            font-weight: 500;
        }

        /* Buttons - Apple style with glow */
        button {
            padding: var(--space-2) var(--space-5);
            font-size: var(--text-subhead);
            font-weight: 600;
            font-family: var(--font-sans);
            background: var(--accent);
            color: white;
            border: none;
            border-radius: var(--radius-md);
            cursor: pointer;
            transition: all var(--duration-base) var(--ease-out);
            box-shadow: 0 2px 8px var(--accent-glow);
            position: relative;
            overflow: hidden;
        }

        button:hover:not(:disabled) {
            background: var(--accent-hover);
            transform: translateY(-1px);
            box-shadow: 0 4px 16px var(--accent-glow);
        }

        button:active:not(:disabled) {
            transform: scale(0.98) translateY(0);
            transition: transform var(--duration-fast) var(--ease-out);
        }

        button:disabled {
            background: rgba(255, 255, 255, 0.08);
            color: var(--text-quaternary);
            cursor: not-allowed;
            box-shadow: none;
            transform: none;
        }

        .btn-row {
            display: flex;
            gap: var(--space-2);
        }
        .btn-row button {
            flex: 1;
            min-height: 48px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: var(--space-1);
            white-space: nowrap;
        }

        /* Fill button row - buttons expand to fill space */
        .btn-row-fill {
            display: flex;
            gap: var(--space-2);
            margin-bottom: var(--space-2);
        }
        .btn-row-fill > button,
        .btn-row-fill > .btn-secondary {
            flex: 1;
            min-height: 44px;
        }

        /* Hidden utility */
        .hidden { display: none !important; }

        .btn-secondary {
            background: var(--glass-bg);
            color: var(--text-secondary);
            border: 1px solid var(--glass-border);
            box-shadow: none;
        }

        .btn-secondary:hover:not(:disabled) {
            background: var(--glass-bg-hover);
            color: var(--text-primary);
            border-color: var(--glass-border-hover);
            box-shadow: none;
            transform: none;
        }

        .btn-secondary:disabled {
            background: rgba(255, 255, 255, 0.03);
            color: var(--text-quaternary);
            border-color: rgba(255, 255, 255, 0.06);
            cursor: not-allowed;
            opacity: 0.5;
            pointer-events: none;
        }

        /* .btn-green removed - use accent color for all primary actions */

        /* Cancel button - subdued until hovered */
        .btn-cancel {
            background: var(--error-bg-hover);
            color: var(--error-muted);
            border: 1px solid var(--error-border);
            box-shadow: none;
        }
        .btn-cancel:hover:not(:disabled) {
            background: var(--error-bg-active);
            color: var(--error);
            border-color: var(--error-border-hover);
            box-shadow: none;
            transform: none;
        }

        /* Cheatsheet buttons */
        .cheat-btn {
            padding: var(--space-2) var(--space-4);
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            border-radius: var(--radius-sm);
            color: var(--text-primary);
            cursor: pointer;
            font-size: var(--text-footnote);
            transition: all var(--duration-fast) var(--ease-out);
            min-height: 44px;
        }
        .cheat-btn:hover {
            background: var(--accent-bg);
            border-color: var(--accent-border-hover);
        }

        /* Textarea container with AI buttons */
        .textarea-container {
            position: relative;
        }

        .textarea-container textarea {
            padding-right: 60px; /* Space for AI buttons column */
        }

        .ai-buttons {
            position: absolute;
            right: var(--space-2);
            top: var(--space-2);
            display: flex;
            flex-direction: column;
            gap: var(--space-1);
        }

        /* Floating AI buttons - 44px touch targets */
        .ai-float-btn {
            width: 44px;
            height: 44px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: var(--radius-sm);
            color: var(--text-tertiary);
            cursor: pointer;
            font-size: var(--text-subhead);
            transition: all var(--duration-fast) var(--ease-out);
        }

        .ai-float-btn:hover {
            background: var(--accent-bg-hover);
            border-color: var(--accent-border);
            color: var(--text-primary);
        }

        /* Status */
        #status {
            text-align: center;
            padding: var(--space-3) var(--space-4);
            margin: var(--space-3) 0;
            border-radius: var(--radius-md);
            display: none;
            font-size: var(--text-subhead);
            font-weight: 500;
            backdrop-filter: blur(10px);
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }

        .generating {
            background: var(--accent-bg);
            border: 1px solid var(--accent-bg-active);
            animation: pulse 1.5s ease-in-out infinite;
            color: var(--accent);
        }

        .success {
            background: var(--success-bg);
            border: 1px solid rgba(48, 209, 88, 0.25);
            color: var(--success);
        }

        .error {
            background: var(--error-bg);
            border: 1px solid var(--error-bg-active);
            color: var(--error);
        }

        /* Progress */
        .progress-container { margin-top: var(--space-2); }

        .progress-bar {
            width: 100%;
            height: 6px;
            background: rgba(255, 255, 255, 0.08);
            border-radius: var(--radius-full);
            overflow: hidden;
        }

        .progress-fill {
            height: 100%;
            background: var(--accent);  /* Solid color - no gradient, no glow */
            border-radius: var(--radius-full);
            transition: width var(--duration-slow) var(--ease-out);
        }

        .progress-text {
            display: flex;
            justify-content: space-between;
            margin-top: var(--space-1);
            font-size: var(--text-caption);
            color: var(--text-tertiary);
            font-family: var(--font-mono);
        }

        /* Result */
        @keyframes resultImageIn {
            from { opacity: 0; transform: scale(0.96) translateY(12px); }
            to { opacity: 1; transform: scale(1) translateY(0); }
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        #result {
            text-align: center;
            margin: var(--space-6) 0;
        }

        #result img {
            max-width: 100%;
            border-radius: var(--radius-xl);
            box-shadow: 0 16px 48px rgba(0, 0, 0, 0.35),
                        0 0 32px rgba(99, 102, 241, 0.08);
            animation: resultImageIn 0.4s var(--ease-spring);
        }

        .result-actions {
            margin-top: var(--space-4);
            display: flex;
            gap: var(--space-2);
            justify-content: center;
            flex-wrap: wrap;
        }

        .result-info {
            margin-top: var(--space-2);
            font-size: var(--text-footnote);
            color: var(--text-tertiary);
        }

        .seed-display {
            font-family: var(--font-mono);
            background: var(--glass-bg);
            padding: var(--space-1) var(--space-2);
            border-radius: var(--radius-sm);
            cursor: pointer;
            font-size: var(--text-caption);
            border: 1px solid var(--glass-border);
            transition: all var(--duration-base) var(--ease-out);
        }

        .seed-display:hover {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-glow);
        }

        /* Batch results grid */
        .batch-results {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: var(--space-3);
            margin-top: var(--space-3);
        }

        .batch-item {
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            border-radius: var(--radius-lg);
            overflow: hidden;
            transition: all var(--duration-base) var(--ease-out);
        }

        .batch-item:hover {
            border-color: var(--glass-border-hover);
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
        }

        .batch-item img {
            width: 100%;
            aspect-ratio: 1;
            object-fit: cover;
            display: block;
        }

        .batch-item-actions {
            display: flex;
            gap: var(--space-1);
            padding: var(--space-2);
            justify-content: center;
        }

        .batch-seed {
            font-size: var(--text-caption);
            font-family: var(--font-mono);
            color: var(--text-quaternary);
            text-align: center;
            padding-bottom: var(--space-2);
        }

        .btn-sm {
            padding: var(--space-1) var(--space-2);
            font-size: var(--text-caption);
            border-radius: var(--radius-sm);
        }

        /* Batch progress indicator */
        .batch-progress {
            display: flex;
            gap: var(--space-2);
            margin-bottom: var(--space-3);
            flex-wrap: wrap;
            justify-content: center;
        }

        .batch-progress-item {
            display: flex;
            align-items: center;
            gap: var(--space-1);
            padding: var(--space-1) var(--space-2);
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            border-radius: var(--radius-md);
            font-size: var(--text-caption);
            color: var(--text-tertiary);
        }

        .batch-progress-item.active {
            border-color: var(--accent);
            color: var(--accent);
            background: var(--accent-bg);
        }

        .batch-progress-item.done {
            border-color: var(--success);
            color: var(--success);
            background: var(--success-bg);
        }

        .batch-progress-item .mini-progress {
            width: 40px;
            height: 4px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 2px;
            overflow: hidden;
        }

        .batch-progress-item .mini-progress-fill {
            height: 100%;
            background: var(--accent);
            transition: width var(--duration-slow) var(--ease-out);
        }

        /* Gallery - ALL items get animations */
        @keyframes galleryItemIn {
            from { opacity: 0; transform: scale(0.95); }
            to { opacity: 1; transform: scale(1); }
        }

        .gallery {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            gap: var(--space-3);
            margin-top: var(--space-4);
        }

        .gallery-item {
            position: relative;
            aspect-ratio: 1;
            border-radius: var(--radius-lg);
            overflow: hidden;
            cursor: pointer;
            border: 1px solid var(--glass-border);
            background: var(--glass-bg);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            /* Animation for ALL gallery items via CSS */
            animation: galleryItemIn 0.35s var(--ease-out) both;
            transition: transform var(--duration-slow) var(--ease-spring),
                        box-shadow var(--duration-slow) var(--ease-out),
                        border-color var(--duration-base) var(--ease-out);
        }

        /* Stagger animation for first 20 items */
        .gallery-item:nth-child(1) { animation-delay: 0.02s; }
        .gallery-item:nth-child(2) { animation-delay: 0.04s; }
        .gallery-item:nth-child(3) { animation-delay: 0.06s; }
        .gallery-item:nth-child(4) { animation-delay: 0.08s; }
        .gallery-item:nth-child(5) { animation-delay: 0.1s; }
        .gallery-item:nth-child(6) { animation-delay: 0.12s; }
        .gallery-item:nth-child(7) { animation-delay: 0.14s; }
        .gallery-item:nth-child(8) { animation-delay: 0.16s; }
        .gallery-item:nth-child(9) { animation-delay: 0.18s; }
        .gallery-item:nth-child(10) { animation-delay: 0.2s; }
        .gallery-item:nth-child(11) { animation-delay: 0.22s; }
        .gallery-item:nth-child(12) { animation-delay: 0.24s; }
        .gallery-item:nth-child(13) { animation-delay: 0.26s; }
        .gallery-item:nth-child(14) { animation-delay: 0.28s; }
        .gallery-item:nth-child(15) { animation-delay: 0.3s; }
        .gallery-item:nth-child(16) { animation-delay: 0.32s; }
        .gallery-item:nth-child(17) { animation-delay: 0.34s; }
        .gallery-item:nth-child(18) { animation-delay: 0.36s; }
        .gallery-item:nth-child(19) { animation-delay: 0.38s; }
        .gallery-item:nth-child(20) { animation-delay: 0.4s; }
        .gallery-item:nth-child(n+21) { animation-delay: 0.42s; }

        .gallery-item:hover {
            transform: scale(1.04) translateY(-2px);
            border-color: var(--accent);
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.25),
                        0 0 20px var(--accent-glow);
        }

        .gallery-item img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            transition: transform var(--duration-slow) var(--ease-out);
        }

        .gallery-item:hover img {
            transform: scale(1.02);
        }

        .gallery-item .gallery-actions {
            position: absolute;
            top: var(--space-2);
            right: var(--space-2);
            display: flex;
            gap: var(--space-1);
            opacity: 0;
            transition: opacity var(--duration-base) var(--ease-out);
        }

        .gallery-item:hover .gallery-actions {
            opacity: 1;
        }

        .gallery-item .favorite-star, .gallery-item .delete-btn {
            font-size: var(--text-footnote);
            background: rgba(0, 0, 0, 0.6);
            backdrop-filter: blur(8px);
            padding: var(--space-2);
            border-radius: var(--radius-sm);
            cursor: pointer;
            transition: all var(--duration-base) var(--ease-spring);
        }

        .gallery-item .favorite-star:hover, .gallery-item .delete-btn:hover {
            transform: scale(1.15);
            background: rgba(0, 0, 0, 0.8);
        }

        .gallery-item .delete-btn {
            opacity: 0;
            transition: opacity var(--duration-base);
        }
        .gallery-item:hover .delete-btn { opacity: 1; }

        .gallery-item .gallery-type-badge {
            position: absolute;
            bottom: var(--space-2);
            left: var(--space-2);
            background: rgba(0, 0, 0, 0.65);
            backdrop-filter: blur(8px);
            padding: var(--space-1) var(--space-2);
            border-radius: var(--radius-sm);
            font-size: var(--text-caption);
            font-weight: 500;
            transition: transform var(--duration-base) var(--ease-out);
        }

        .gallery-item:hover .gallery-type-badge {
            transform: translateY(-2px);
        }

        .gallery-item .gallery-info {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background: linear-gradient(transparent, rgba(0, 0, 0, 0.8));
            padding: var(--space-6) var(--space-2) var(--space-2);
            opacity: 0;
            transition: opacity var(--duration-base) var(--ease-out);
        }

        .gallery-item:hover .gallery-info {
            opacity: 1;
        }

        .gallery-info-text {
            font-size: var(--text-caption);
            color: var(--text-secondary);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .gallery-empty {
            grid-column: 1 / -1;
            text-align: center;
            padding: var(--space-8);
            color: var(--text-tertiary);
        }

        .gallery-count {
            font-size: var(--text-footnote);
            color: var(--text-quaternary);
            margin-left: var(--space-2);
        }

        /* Modal */
        @keyframes modalIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        @keyframes modalImageIn {
            from { opacity: 0; transform: scale(0.92); }
            to { opacity: 1; transform: scale(1); }
        }

        .modal {
            display: none;
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: rgba(0, 0, 0, 0.9);
            backdrop-filter: blur(20px);
            z-index: 1000;
            justify-content: center;
            align-items: center;
        }

        .modal.active {
            display: flex;
            animation: modalIn 0.25s var(--ease-out);
        }

        .modal img {
            max-width: 90%;
            max-height: 90%;
            border-radius: var(--radius-xl);
            box-shadow: 0 24px 80px rgba(0, 0, 0, 0.5);
            animation: modalImageIn 0.35s var(--ease-spring);
        }

        .modal-close {
            position: absolute;
            top: var(--space-6);
            right: var(--space-8);
            font-size: var(--text-title2);
            color: var(--text-tertiary);
            cursor: pointer;
            transition: all var(--duration-base) var(--ease-out);
            width: 44px;
            height: 44px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: var(--radius-full);
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
        }

        .modal-close:hover {
            color: var(--text-primary);
            background: var(--glass-bg-hover);
            transform: scale(1.05);
        }

        /* Modal content panel - standardized */
        .modal-panel {
            background: rgba(30, 30, 50, 0.98);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            padding: var(--space-5);
            border-radius: var(--radius-lg);
            max-width: 600px;
            max-height: 80vh;
            overflow-y: auto;
        }
        .modal-panel-wide { max-width: 650px; max-height: 85vh; }
        .modal-panel-compact { max-width: 500px; }

        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: var(--space-4);
        }
        .modal-header h3 { margin: 0; font-size: var(--text-title3); }

        .modal-close-btn {
            background: none;
            border: none;
            color: var(--text-tertiary);
            font-size: var(--text-title3);
            cursor: pointer;
            padding: var(--space-2);
            margin: calc(-1 * var(--space-2));
            line-height: 1;
            transition: color var(--duration-fast);
        }
        .modal-close-btn:hover { color: var(--text-primary); }

        .modal-description {
            font-size: var(--text-footnote);
            color: var(--text-secondary);
            margin-bottom: var(--space-4);
        }

        .modal-grid-3 {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: var(--space-3);
        }

        .modal-card {
            background: rgba(255, 255, 255, 0.05);
            padding: var(--space-3);
            border-radius: var(--radius-md);
        }

        .modal-card h4 {
            margin: 0 0 var(--space-2) 0;
            font-size: var(--text-footnote);
            color: var(--accent);
        }

        .modal-card-content {
            font-size: var(--text-caption);
            line-height: 1.8;
        }

        .modal-highlight {
            padding: var(--space-3);
            background: var(--accent-bg);
            border: 1px solid var(--accent-bg-active);
            border-radius: var(--radius-md);
            margin-top: var(--space-4);
        }

        .modal-highlight h4 {
            margin: 0 0 var(--space-2) 0;
            font-size: var(--text-footnote);
        }

        .modal-btn-row {
            display: flex;
            flex-wrap: wrap;
            gap: var(--space-2);
        }

        .modal-chip {
            padding: var(--space-2) var(--space-3);
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: var(--radius-sm);
            color: var(--text-primary);
            cursor: pointer;
            font-size: var(--text-caption);
            transition: all var(--duration-fast);
        }
        .modal-chip:hover {
            background: rgba(255, 255, 255, 0.15);
        }

        .modal-actions {
            margin-top: var(--space-4);
            text-align: center;
        }

        .gallery-compact {
            grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
        }

        /* Form input (used in modals) */
        .form-input {
            width: 100%;
            padding: var(--space-3);
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: var(--radius-md);
            color: var(--text-primary);
            font-size: var(--text-subhead);
            font-family: var(--font-sans);
            box-sizing: border-box;
            transition: all var(--duration-base) var(--ease-out);
        }
        .form-input:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-glow);
        }

        .form-label {
            font-size: var(--text-caption);
            color: var(--text-secondary);
            margin-bottom: var(--space-2);
            display: block;
        }

        .form-group {
            margin-bottom: var(--space-4);
        }

        .form-preview {
            padding: var(--space-3);
            background: var(--accent-bg);
            border: 1px solid var(--accent-bg-active);
            border-radius: var(--radius-md);
            margin-bottom: var(--space-4);
        }
        .form-preview-label {
            font-size: var(--text-caption);
            color: var(--text-tertiary);
            margin-bottom: var(--space-1);
        }
        .form-preview-text {
            font-size: var(--text-subhead);
            color: var(--text-primary);
            min-height: 20px;
        }

        /* Code highlight */
        code {
            background: rgba(255, 255, 255, 0.08);
            padding: var(--space-1) var(--space-2);
            border-radius: var(--radius-sm);
            font-family: var(--font-mono);
            font-size: 0.9em;
        }

        /* Result container - for generated content */
        .result-container {
            background: var(--glass-bg);
            border-radius: var(--radius-lg);
            padding: var(--space-5);
            margin-top: var(--space-4);
        }

        .result-video {
            width: 100%;
            border-radius: var(--radius-md);
        }

        .result-actions {
            margin-top: var(--space-4);
            display: flex;
            gap: var(--space-2);
        }

        .result-actions > * {
            flex: 1;
            text-align: center;
            text-decoration: none;
        }

        .result-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: var(--space-4);
            margin-bottom: var(--space-4);
        }

        .result-grid-item {
            text-align: center;
        }

        .result-grid-label {
            font-size: var(--text-footnote);
            color: var(--text-tertiary);
            margin-bottom: var(--space-2);
        }

        .result-grid-image {
            width: 100%;
            border-radius: var(--radius-md);
            border: 1px solid var(--glass-border);
        }

        .result-grid-image-accent {
            border-color: var(--accent);
        }

        .result-spinner {
            text-align: center;
            padding: var(--space-5);
        }

        .result-spinner .spinner {
            width: 32px;
            height: 32px;
            border: 3px solid var(--glass-border);
            border-top-color: var(--accent);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }

        .result-spinner p {
            margin-top: var(--space-3);
            color: var(--text-secondary);
        }

        .result-error {
            color: var(--error);
        }

        /* Examples */
        .examples {
            margin-top: var(--space-6);
            padding: var(--space-5);
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            border-radius: var(--radius-lg);
            animation: fadeInUp 0.4s var(--ease-out) 0.3s both;
        }

        .examples h3 {
            margin: 0 0 var(--space-3) 0;
            font-size: var(--text-subhead);
            color: var(--text-secondary);
            font-weight: 600;
        }

        /* Prompt ideas grid - replaces inline styles */
        .prompt-ideas-grid {
            display: flex;
            gap: var(--space-2);
            flex-wrap: wrap;
        }

        .example-btn {
            display: inline-flex;
            align-items: center;
            padding: var(--space-3) var(--space-4);
            /* margin removed - gap on parent handles spacing */
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid var(--glass-border);
            border-radius: var(--radius-full);
            cursor: pointer;
            font-size: var(--text-footnote);
            color: var(--text-secondary);
            transition: all var(--duration-base) var(--ease-out);
            min-height: 44px;  /* HIG minimum - NON-NEGOTIABLE */
        }

        .example-btn:hover {
            background: rgba(255, 255, 255, 0.12);
            border-color: var(--glass-border-hover);
            color: var(--text-primary);
        }

        .example-btn:active {
            transform: scale(0.97);
        }

        /* Advanced toggle */
        .advanced-toggle {
            color: var(--text-tertiary);
            cursor: pointer;
            font-size: var(--text-footnote);
            margin-bottom: var(--space-2);
            display: inline-flex;
            align-items: center;
            gap: var(--space-1);
            transition: color var(--duration-base);
        }

        .advanced-toggle:hover { color: var(--text-secondary); }
        .advanced-section { display: none; }
        .advanced-section.show {
            display: block;
            animation: fadeIn 0.2s var(--ease-out);
        }

        /* Option hints - subtle helper text */
        .option-hint {
            display: block;
            font-size: var(--text-caption);
            color: var(--text-quaternary);
            margin-bottom: var(--space-2);
            line-height: 1.4;
            font-weight: 400;
        }

        /* Upload area */
        .upload-area {
            border: 2px dashed var(--glass-border);
            border-radius: var(--radius-lg);
            padding: var(--space-8);
            text-align: center;
            cursor: pointer;
            transition: all var(--duration-slow) var(--ease-spring);
            margin-bottom: var(--space-3);
            color: var(--text-tertiary);
            background: rgba(0, 0, 0, 0.1);
        }

        .upload-area:hover {
            border-color: var(--accent);
            color: var(--text-secondary);
            background: var(--accent-bg);
            transform: translateY(-2px);
        }

        .upload-area.dragover {
            border-color: var(--accent);
            background: var(--accent-bg);
            box-shadow: 0 0 24px var(--accent-glow);
            transform: scale(1.01);
        }

        .upload-preview {
            max-width: 200px;
            max-height: 200px;
            margin-top: var(--space-2);
            border-radius: var(--radius-lg);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
        }

        .upload-area-compact {
            padding: var(--space-4);
            min-height: 120px;
            margin-bottom: var(--space-4);
        }
        .upload-area-compact .upload-preview {
            max-height: 100px;
        }

        /* Filter tabs */
        .filter-tabs {
            display: flex;
            gap: var(--space-2);
            margin-bottom: var(--space-3);
            flex-wrap: wrap;
        }

        .filter-tab {
            padding: var(--space-2) var(--space-4);
            background: var(--glass-bg);
            border: 2px solid var(--glass-border);  /* Match preset-btn */
            border-radius: var(--radius-full);
            cursor: pointer;
            font-size: var(--text-footnote);
            font-weight: 500;
            color: var(--text-secondary);
            transition: all var(--duration-base) var(--ease-out);
            min-height: 44px;  /* HIG minimum - NON-NEGOTIABLE */
            display: inline-flex;
            align-items: center;
        }

        .filter-tab:hover {
            background: var(--glass-bg-hover);
            border-color: var(--glass-border-hover);
            color: var(--text-primary);
        }

        .filter-tab:active {
            transform: scale(0.97);
        }

        .filter-tab.active {
            background: var(--accent-bg-active);
            border-color: var(--accent);
            color: var(--text-primary);
            font-weight: 600;
            box-shadow: 0 0 0 2px var(--accent-ring);
        }

        /* Presets - Pill buttons */
        .presets {
            display: flex;
            gap: var(--space-1);
            margin-bottom: var(--space-3);
            overflow-x: auto;
            padding-bottom: var(--space-1);
            -webkit-overflow-scrolling: touch;
            scrollbar-width: thin;
        }
        .presets::-webkit-scrollbar { height: var(--space-1); }
        .presets::-webkit-scrollbar-track { background: transparent; }
        .presets::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.2); border-radius: 2px; }

        /* Preset buttons - unified with model-btn active states */
        .preset-btn {
            padding: var(--space-2) var(--space-4);
            background: var(--glass-bg);
            border: 2px solid var(--glass-border);
            border-radius: var(--radius-md);
            cursor: pointer;
            font-size: var(--text-footnote);
            color: var(--text-secondary);
            transition: all var(--duration-base) var(--ease-out);
            white-space: nowrap;
            flex: 1;
            min-width: 100px;
            min-height: 44px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
        }

        .preset-btn:hover {
            background: var(--glass-bg-hover);
            border-color: var(--glass-border-hover);
            color: var(--text-primary);
        }

        .preset-btn:active {
            transform: scale(0.98);
        }

        .preset-btn.active {
            background: var(--accent-bg-active);
            border-color: var(--accent);
            color: var(--text-primary);
            font-weight: 600;
            box-shadow: 0 0 0 2px var(--accent-ring);
        }

        /* Model selector container */
        .model-selector {
            display: flex;
            gap: var(--space-2);
            margin-bottom: var(--space-4);
        }

        /* Model selector buttons - Unified with accent color */
        .model-btn {
            flex: 1;
            padding: var(--space-3) var(--space-4);
            background: var(--glass-bg);
            border: 2px solid var(--glass-border);
            border-radius: var(--radius-md);
            color: var(--text-secondary);
            cursor: pointer;
            font-size: var(--text-footnote);
            text-align: center;
            transition: all var(--duration-base) var(--ease-out);
            min-height: 56px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            gap: var(--space-1);
        }

        .model-btn:hover {
            background: var(--glass-bg-hover);
            border-color: var(--glass-border-hover);
            color: var(--text-primary);
        }

        .model-btn.active {
            background: var(--accent-bg-active);
            border-color: var(--accent);
            color: var(--text-primary);
            box-shadow: 0 0 0 2px var(--accent-ring);
        }

        .model-btn .model-title {
            font-weight: 600;
            font-size: var(--text-footnote);
        }

        .model-btn .model-subtitle {
            font-size: var(--text-caption);
            opacity: 0.7;
        }

        /* Mode buttons - Same style as model buttons */
        .mode-btn {
            flex: 1;
            padding: var(--space-3) var(--space-4);
            background: var(--glass-bg);
            border: 2px solid var(--glass-border);
            border-radius: var(--radius-md);
            color: var(--text-secondary);
            cursor: pointer;
            font-size: var(--text-footnote);
            text-align: center;
            transition: all var(--duration-base) var(--ease-out);
            min-height: 44px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: var(--space-1);
        }

        .mode-btn:hover {
            background: var(--glass-bg-hover);
            border-color: var(--glass-border-hover);
            color: var(--text-primary);
        }

        .mode-btn.active {
            background: var(--accent-bg-active);
            border-color: var(--accent);
            color: var(--text-primary);
            box-shadow: 0 0 0 2px var(--accent-ring);
        }

        /* Upscale buttons - Same consistent style */
        .upscale-btn {
            flex: 1;
            padding: var(--space-3);
            background: var(--glass-bg);
            border: 2px solid var(--glass-border);
            border-radius: var(--radius-md);
            color: var(--text-secondary);
            cursor: pointer;
            text-align: center;
            transition: all var(--duration-base) var(--ease-out);
        }

        .upscale-btn:hover {
            background: var(--glass-bg-hover);
            border-color: var(--glass-border-hover);
            color: var(--text-primary);
        }

        .upscale-btn.active {
            background: var(--accent-bg-active);
            border-color: var(--accent);
            color: var(--text-primary);
            box-shadow: 0 0 0 2px var(--accent-ring);
        }

        .upscale-hint {
            font-size: var(--text-caption);
            opacity: 0.7;
        }

        /* Edit mode section layout */
        .edit-mode-section {
            margin: var(--space-4) 0;
        }

        .edit-mode-buttons {
            display: flex;
            gap: var(--space-2);
            margin-top: var(--space-3);
            flex-wrap: wrap;
        }

        .upscale-controls {
            display: none;
            margin-top: var(--space-3);
        }

        .upscale-controls.show {
            display: block;
        }

        .upscale-options {
            display: flex;
            gap: var(--space-3);
        }

        /* Button divider */
        .btn-divider {
            border-left: 1px solid var(--glass-border);
            margin: 0 var(--space-1);
            height: auto;
            align-self: stretch;
        }

        /* Tertiary button - text style with subtle background */
        .btn-tertiary {
            padding: var(--space-3) var(--space-4);
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            border-radius: var(--radius-md);
            color: var(--text-secondary);
            cursor: pointer;
            font-size: var(--text-footnote);
            transition: all var(--duration-base) var(--ease-out);
            min-height: 44px;
        }

        .btn-tertiary:hover {
            background: var(--glass-bg-hover);
            border-color: var(--glass-border-hover);
            color: var(--text-primary);
        }

        .btn-tertiary.btn-warning {
            background: rgba(255, 159, 10, 0.1);
            border-color: rgba(255, 159, 10, 0.3);
            color: var(--warning);
        }

        .btn-tertiary.btn-warning:hover {
            background: rgba(255, 159, 10, 0.2);
            border-color: rgba(255, 159, 10, 0.5);
        }

        .btn-tertiary.btn-sm {
            padding: var(--space-2) var(--space-3);
            font-size: var(--text-caption);
            min-height: auto;
        }

        /* Examples section header */
        .examples-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: var(--space-2);
        }

        .examples-header h3 {
            margin: 0;
            font-size: var(--text-subhead);
            font-weight: 600;
            color: var(--text-secondary);
        }

        /* Settings page styles */
        .settings-card {
            background: var(--glass-bg);
            padding: var(--space-5);
            border-radius: var(--radius-lg);
            margin-bottom: var(--space-5);
            border: 1px solid var(--glass-border);
        }

        .settings-label {
            font-size: var(--text-body);
            margin-bottom: var(--space-4);
            display: block;
        }

        .settings-item {
            display: flex;
            align-items: center;
            gap: var(--space-3);
            padding: var(--space-4);
            background: var(--glass-bg);
            border-radius: var(--radius-md);
            border: 1px solid var(--glass-border);
        }

        .settings-item.active {
            background: var(--accent-bg-subtle);
            border-color: var(--accent-bg-active);
        }

        .settings-icon {
            font-size: var(--text-title3);
        }

        .settings-item-title {
            font-weight: 600;
            color: var(--text-primary);
        }

        .settings-item-desc {
            font-size: var(--text-footnote);
            color: var(--text-tertiary);
            margin-top: var(--space-1);
        }

        .settings-models {
            display: flex;
            flex-direction: column;
            gap: var(--space-2);
        }

        .settings-model-item {
            display: flex;
            align-items: baseline;
            gap: var(--space-3);
            font-size: var(--text-footnote);
            color: var(--text-secondary);
            padding: var(--space-3) var(--space-2);
            border-radius: var(--radius-sm);
            transition: background var(--duration-fast) var(--ease-out);
        }

        .settings-model-item:hover {
            background: rgba(255, 255, 255, 0.03);
        }

        .settings-model-item:last-child {
            margin-bottom: 0;
            padding-bottom: var(--space-2);
        }

        .settings-model-icon {
            flex-shrink: 0;
        }

        .settings-model-label {
            font-weight: 600;
            color: var(--text-primary);
            white-space: nowrap;
        }

        .settings-model-value {
            color: var(--text-tertiary);
            font-family: var(--font-mono);
            font-size: var(--text-caption);
        }

        .settings-status {
            padding: var(--space-4);
            background: var(--success-bg);
            border: 1px solid rgba(48, 209, 88, 0.3);
            border-radius: var(--radius-md);
            display: none;
            margin-top: var(--space-4);
            color: var(--success);
            font-weight: 500;
        }

        /* Status card - for model status sections */
        .status-card {
            margin-top: var(--space-4);
            padding: var(--space-5);
            background: var(--success-bg);
            border: 1px solid rgba(48, 209, 88, 0.3);
            border-radius: var(--radius-lg);
        }

        .status-card h4 {
            margin: 0 0 var(--space-3) 0;
            font-size: var(--text-body);
            font-weight: 600;
        }

        .status-card p {
            font-size: var(--text-footnote);
            color: var(--text-secondary);
            margin-bottom: var(--space-3);
        }

        .model-list {
            font-size: var(--text-caption);
            color: var(--text-tertiary);
        }

        .model-list-item {
            margin-bottom: var(--space-2);
        }

        .model-list-item strong {
            color: var(--text-secondary);
        }

        .model-list-item .model-file {
            margin-left: var(--space-4);
            margin-top: var(--space-1);
        }

        .model-list-item code {
            background: rgba(255, 255, 255, 0.08);
            padding: var(--space-1) var(--space-2);
            border-radius: var(--radius-sm);
            font-family: var(--font-mono);
        }

        /* History dropdown */
        @keyframes dropdownOpen {
            from { opacity: 0; transform: scaleY(0.95) translateY(-8px); }
            to { opacity: 1; transform: scaleY(1) translateY(0); }
        }

        .history-container { position: relative; margin-bottom: var(--space-3); }

        .history-btn {
            padding: var(--space-2) var(--space-3);
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            border-radius: var(--radius-md);
            cursor: pointer;
            font-size: var(--text-footnote);
            display: inline-flex;
            align-items: center;
            gap: var(--space-1);
            color: var(--text-secondary);
            transition: all var(--duration-base) var(--ease-out);
        }

        .history-btn:hover {
            background: var(--glass-bg-hover);
            color: var(--text-primary);
        }

        .history-dropdown {
            position: absolute;
            top: calc(100% + var(--space-1));
            left: 0; right: 0;
            background: rgba(25, 25, 40, 0.95);
            backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            border-radius: var(--radius-lg);
            max-height: 320px;
            overflow: hidden;
            display: none;
            z-index: 100;
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.35);
        }
        .history-search {
            padding: var(--space-2) var(--space-3);
            border-bottom: 1px solid var(--glass-border);
            position: sticky;
            top: 0;
            background: rgba(25, 25, 40, 0.98);
        }
        .history-search input {
            width: 100%;
            padding: var(--space-2) var(--space-3);
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            border-radius: var(--radius-md);
            color: var(--text-primary);
            font-size: var(--text-footnote);
        }
        .history-search input::placeholder { color: var(--text-quaternary); }
        .history-search input:focus { outline: none; border-color: var(--accent); }
        .history-list {
            max-height: 260px;
            overflow-y: auto;
        }
        .history-empty {
            padding: var(--space-4);
            text-align: center;
            color: var(--text-quaternary);
            font-size: var(--text-footnote);
        }

        .history-dropdown.show {
            display: block;
            animation: dropdownOpen 0.2s var(--ease-spring);
        }

        .history-item {
            padding: var(--space-3) var(--space-4);
            cursor: pointer;
            border-bottom: 1px solid var(--glass-border);
            font-size: var(--text-subhead);
            color: var(--text-secondary);
            transition: all var(--duration-base);
        }

        .history-item:hover {
            background: var(--glass-bg-hover);
            color: var(--text-primary);
        }

        .history-item:last-child { border-bottom: none; }

        .history-meta {
            font-size: var(--text-caption);
            color: var(--text-quaternary);
            margin-top: var(--space-1);
            font-family: var(--font-mono);
        }

        .history-item-content {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .history-item-text {
            flex: 1;
            cursor: pointer;
        }

        .history-item-delete {
            cursor: pointer;
            padding: var(--space-2);
            opacity: 0.6;
            transition: opacity var(--duration-fast);
            font-size: var(--text-body);
        }

        .history-item-delete:hover {
            opacity: 1;
        }

        /* Batch options */
        .batch-row {
            display: flex;
            align-items: center;
            gap: var(--space-2);
            margin-bottom: var(--space-3);
        }

        .batch-label { font-size: var(--text-footnote); color: var(--text-tertiary); }

        .batch-input {
            width: 80px;
            min-width: 80px;
            max-width: 80px;
            text-align: center;
            font-family: var(--font-mono);
        }

        /* Generation Queue */
        .queue-container {
            margin-top: var(--space-3);
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            overflow: hidden;
            display: none;
        }
        .queue-container.active { display: block; }
        .queue-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: var(--space-2) var(--space-3);
            background: var(--bg-tertiary);
            border-bottom: 1px solid var(--border);
        }
        .queue-title {
            font-size: var(--text-footnote);
            font-weight: 600;
            color: var(--text-secondary);
        }
        .queue-count {
            font-size: var(--text-caption);
            font-family: var(--font-mono);
            color: var(--accent);
            background: var(--accent-bg);
            padding: var(--space-1) var(--space-2);
            border-radius: var(--radius-full);
        }
        .queue-list {
            max-height: 200px;
            overflow-y: auto;
        }
        .queue-item {
            display: flex;
            align-items: center;
            gap: var(--space-2);
            padding: var(--space-2) var(--space-3);
            border-bottom: 1px solid var(--border);
            font-size: var(--text-footnote);
        }
        .queue-item:last-child { border-bottom: none; }
        .queue-item.processing {
            background: var(--accent-bg);
        }
        .queue-item-status {
            width: 20px;
            text-align: center;
        }
        .queue-item-prompt {
            flex: 1;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            color: var(--text-secondary);
        }
        .queue-item-settings {
            font-size: var(--text-caption);
            font-family: var(--font-mono);
            color: var(--text-quaternary);
        }
        .queue-item-remove {
            cursor: pointer;
            color: var(--text-quaternary);
            transition: color var(--duration-fast);
        }
        .queue-item-remove:hover { color: var(--error); }
        .queue-actions {
            display: flex;
            gap: var(--space-2);
            padding: var(--space-2) var(--space-3);
            border-top: 1px solid var(--border);
            background: var(--bg-tertiary);
        }
        .queue-btn {
            flex: 1;
            padding: var(--space-1) var(--space-2);
            font-size: var(--text-caption);
            border-radius: var(--radius-md);
            cursor: pointer;
            border: none;
            transition: all var(--duration-fast);
        }
        .queue-btn-start {
            background: var(--accent);
            color: white;
        }
        .queue-btn-start:hover { background: var(--accent-hover); }
        .queue-btn-start:disabled {
            background: var(--bg-tertiary);
            color: var(--text-quaternary);
            cursor: not-allowed;
        }
        .queue-btn-clear {
            background: var(--bg-secondary);
            color: var(--text-tertiary);
            border: 1px solid var(--border);
        }
        .queue-btn-clear:hover {
            background: var(--error);
            color: white;
            border-color: var(--error);
        }

        /* Split Compare Mode (Generate Tab) */
        .split-compare-container {
            display: none;
            margin-bottom: var(--space-4);
            padding: var(--space-4);
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            border-radius: var(--radius-xl);
            animation: fadeInUp 0.3s var(--ease-out);
        }

        .split-compare-header {
            display: flex;
            align-items: center;
            gap: var(--space-3);
            margin-bottom: var(--space-4);
            padding-bottom: var(--space-3);
            border-bottom: 1px solid var(--glass-border);
        }

        .split-compare-title {
            font-size: var(--text-body);
            font-weight: 600;
            color: var(--text-primary);
        }

        .split-compare-hint {
            font-size: var(--text-footnote);
            color: var(--text-tertiary);
            flex: 1;
        }

        .split-compare-panels {
            display: grid;
            grid-template-columns: 1fr auto 1fr;
            gap: var(--space-3);
        }

        .split-panel {
            display: flex;
            flex-direction: column;
            gap: var(--space-3);
        }

        .split-panel-header {
            display: flex;
            align-items: center;
            gap: var(--space-2);
        }

        .split-panel-label {
            width: 28px;
            height: 28px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: var(--accent);
            color: white;
            font-weight: 700;
            font-size: var(--text-footnote);
            border-radius: var(--radius-sm);
        }

        .split-panel-title {
            font-size: var(--text-footnote);
            font-weight: 500;
            color: var(--text-secondary);
        }

        .split-prompt {
            height: 80px;
            resize: none;
        }

        .split-generate-btn {
            width: 100%;
        }

        .split-result {
            aspect-ratio: 1;
            background: rgba(0, 0, 0, 0.2);
            border: 1px solid var(--glass-border);
            border-radius: var(--radius-lg);
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }

        .split-result img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            animation: resultImageIn 0.4s var(--ease-spring);
        }

        .split-placeholder {
            color: var(--text-quaternary);
            font-size: var(--text-footnote);
        }

        .split-divider {
            width: 1px;
            background: var(--glass-border);
            margin: 0 var(--space-2);
        }

        /* Compare Mode Container (Gallery) */
        .compare-mode-container {
            display: none;
            margin-bottom: var(--space-4);
            padding: var(--space-4);
            background: var(--accent-bg);
            border: 1px solid var(--accent-bg-active);
            border-radius: var(--radius-lg);
        }

        .compare-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: var(--space-3);
        }

        .compare-title {
            font-size: var(--text-footnote);
            color: var(--text-secondary);
        }

        .compare-exit-btn {
            padding: var(--space-2) var(--space-3);
            background: var(--error-bg-hover);
            color: var(--error);
            border: 1px solid var(--error-border);
            font-size: var(--text-footnote);
            box-shadow: none;
        }

        .compare-exit-btn:hover {
            background: var(--error-bg-active);
        }

        .compare-slots {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: var(--space-3);
        }

        .compare-slot {
            aspect-ratio: 1;
            border: 2px dashed var(--glass-border);
            border-radius: var(--radius-lg);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            color: var(--text-quaternary);
            font-size: var(--text-footnote);
            overflow: hidden;
            background: rgba(0, 0, 0, 0.15);
            transition: all var(--duration-base) var(--ease-out);
        }

        .compare-slot.filled {
            border-style: solid;
            border-color: var(--accent);
        }

        .compare-slot img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            border-radius: calc(var(--radius-lg) - 2px);
        }

        .compare-slot .compare-label {
            position: absolute;
            bottom: var(--space-2);
            left: var(--space-2);
            right: var(--space-2);
            background: rgba(0, 0, 0, 0.7);
            backdrop-filter: blur(8px);
            padding: var(--space-1) var(--space-2);
            border-radius: var(--radius-sm);
            font-size: var(--text-caption);
            color: var(--text-secondary);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        /* Comparison view (result page) */
        .comparison-title {
            text-align: center;
            margin-bottom: var(--space-4);
            font-size: var(--text-title3);
            font-weight: 600;
        }

        .comparison {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: var(--space-4);
            margin-top: var(--space-4);
        }

        .comparison-col { text-align: center; }

        .comparison-col img {
            max-width: 100%;
            border-radius: var(--radius-lg);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
            transition: transform var(--duration-base) var(--ease-out);
        }

        .comparison-col img:hover {
            transform: scale(1.02);
        }

        .comparison-label {
            margin-bottom: var(--space-2);
            font-size: var(--text-footnote);
            color: var(--text-secondary);
            font-weight: 500;
        }

        /* Refiner buttons */
        .refine-btn {
            font-size: var(--text-footnote);
            padding: var(--space-2) var(--space-3);
        }

        /* Scrollbar styling - Thin Apple style */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.15);
            border-radius: var(--radius-full);
            transition: background var(--duration-base);
        }
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.25);
        }

        /* Selection */
        ::selection {
            background: var(--accent);
            color: white;
        }

        /* Focus ring animation */
        *:focus-visible {
            outline: none;
            box-shadow: 0 0 0 3px var(--accent-glow);
            transition: box-shadow var(--duration-base) var(--ease-out);
        }
    </style>
    <!-- Model Viewer for 3D GLB display -->
    <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.3.0/model-viewer.min.js"></script>
</head>
<body>
    <!-- Toast Container -->
    <div id="toastContainer" class="toast-container"></div>

    <h1>Qwen Image Generator</h1>
    <p class="subtitle">Powered by Qwen-Image-2512 on your Mac <span class="connection-status"><span id="statusDot" class="status-dot checking"></span><span id="statusText">Checking...</span></span></p>

    <div class="tabs">
        <button class="tab active" onclick="showTab('generate')">Generate</button>
        <button class="tab" onclick="showTab('edit')">Edit</button>
        <button class="tab" onclick="showTab('video')">Video</button>
        <button class="tab" onclick="showTab('audio')">Audio</button>
        <button class="tab" onclick="showTab('3d')">3D</button>
        <button class="tab" onclick="showTab('gallery')">Gallery</button>
        <button class="tab" onclick="showTab('settings')">Settings</button>
    </div>

    <!-- Generate Tab -->
    <div id="tab-generate" class="tab-content active">
        <div class="input-section">
            <!-- Model Selection -->
            <label>Model</label>
            <div class="model-selector">
                <button type="button" id="imageModelQwen" class="model-btn active" onclick="selectImageModel('qwen')">
                    <span class="model-title">Qwen</span>
                    <span class="model-subtitle">Lightning/Quality</span>
                </button>
                <button type="button" id="imageModelZImage" class="model-btn" onclick="selectImageModel('zimage')">
                    <span class="model-title">Z-Image Turbo</span>
                    <span class="model-subtitle">Fast Photorealistic</span>
                </button>
            </div>

            <!-- Quick Presets (Qwen only) -->
            <div id="qwenPresets">
                <label>Quick Presets</label>
                <div class="presets">
                    <span class="preset-btn active" onclick="applyPreset('quick')" data-preset="quick">Fast</span>
                    <span class="preset-btn" onclick="applyPreset('quality')" data-preset="quality">Quality</span>
                    <span class="preset-btn" onclick="applyPreset('portrait')" data-preset="portrait">Portrait</span>
                    <span class="preset-btn" onclick="applyPreset('landscape')" data-preset="landscape">Landscape</span>
                </div>
            </div>

            <label for="prompt">Describe your image</label>
            <div class="textarea-container">
                <textarea id="prompt" placeholder="A majestic dragon flying over mountains at sunset..."></textarea>
                <div class="ai-buttons">
                    <button type="button" onclick="refineLocal('refine')" title="Refine with AI" class="ai-float-btn">AI</button>
                    <button type="button" onclick="refineLocal('expand')" title="Expand prompt" class="ai-float-btn">+</button>
                    <button type="button" onclick="refineLocal('style')" title="Add style" class="ai-float-btn">S</button>
                </div>
            </div>

            <div class="advanced-toggle" onclick="toggleAdvanced()">Advanced Options</div>
            <div class="advanced-section" id="advancedSection">
                <!-- Prompt History -->
                <div class="option-group" style="margin-bottom: var(--space-3);">
                    <label>Prompt History</label>
                    <span class="option-hint">Click to reuse a previous prompt</span>
                    <div class="history-container" style="margin-top: var(--space-1); display: inline-block;">
                        <div class="history-btn" onclick="toggleHistory()">Recent Prompts <span id="historyCount">(0)</span></div>
                        <div class="history-dropdown" id="historyDropdown"></div>
                    </div>
                </div>

                <div class="option-group" style="margin-bottom: var(--space-3);">
                    <label for="negativePrompt">Negative Prompt</label>
                    <span class="option-hint">Describe what you DON'T want in the image</span>
                    <textarea id="negativePrompt" placeholder="e.g., blurry, ugly, distorted, low quality" class="textarea-short"></textarea>
                </div>

                <div class="options-row">
                    <div class="option-group">
                        <label for="seedInput">Seed</label>
                        <input type="number" id="seedInput" placeholder="Random" title="Same seed = same image">
                    </div>
                    <div class="option-group">
                        <label for="sampler">Sampler</label>
                        <select id="sampler" title="Algorithm for generating">
                            <option value="euler" selected>euler</option>
                            <option value="euler_ancestral">euler_a</option>
                            <option value="dpmpp_2m">dpm++ 2m</option>
                            <option value="dpmpp_sde">dpm++ sde</option>
                            <option value="heun">heun</option>
                        </select>
                    </div>
                    <div class="option-group">
                        <label for="scheduler">Scheduler</label>
                        <select id="scheduler" title="Controls noise reduction">
                            <option value="normal" selected>normal</option>
                            <option value="karras">karras</option>
                            <option value="exponential">exp</option>
                            <option value="simple">simple</option>
                        </select>
                    </div>
                    <div class="option-group" style="flex: 0 0 auto; min-width: auto;">
                        <label>Batch</label>
                        <div style="display: flex; align-items: center; gap: var(--space-2);">
                            <input type="number" id="batchSize" class="batch-input" value="1" min="1" max="4" title="Generate multiple at once">
                        </div>
                    </div>
                </div>
            </div>

            <div class="options-row">
                <div class="option-group" id="modeSelector">
                    <label>Mode</label>
                    <select id="mode" onchange="updateEstimate(); clearPresetHighlight();">
                        <option value="lightning">Lightning</option>
                        <option value="normal">Quality</option>
                    </select>
                </div>
                <div class="option-group">
                    <label>Resolution</label>
                    <select id="resolution" onchange="updateEstimate(); clearPresetHighlight();">
                        <option value="512">512px</option>
                        <option value="768">768px</option>
                        <option value="1024">1024px</option>
                        <option value="1536">1536px</option>
                    </select>
                </div>
                <div class="option-group">
                    <label>Aspect</label>
                    <select id="aspect" onchange="clearPresetHighlight();">
                        <option value="square">Square</option>
                        <option value="landscape">Landscape</option>
                        <option value="portrait">Portrait</option>
                    </select>
                </div>
            </div>
            <div class="time-estimate" id="timeEstimate">Estimated: ~1 minute</div>

            <div class="btn-row">
                <button id="generateBtn" onclick="generate()">Generate</button>
                <button id="cancelBtn" class="btn-cancel" onclick="cancelGeneration()" style="display:none;">Cancel</button>
                <button class="btn-secondary" id="addToQueueBtn" onclick="addToQueue()">Queue</button>
                <button class="btn-secondary" id="regenerateBtn" onclick="regenerate()" disabled>Redo</button>
                <button class="btn-secondary" id="compareBtn" onclick="toggleSplitCompare()">Compare</button>
            </div>

            <!-- Generation Queue -->
            <div id="queueContainer" class="queue-container">
                <div class="queue-header">
                    <span class="queue-title">Generation Queue</span>
                    <span class="queue-count" id="queueCount">0</span>
                </div>
                <div class="queue-list" id="queueList"></div>
                <div class="queue-actions">
                    <button class="queue-btn queue-btn-start" id="queueStartBtn" onclick="processQueue()" disabled>Start Queue</button>
                    <button class="queue-btn queue-btn-clear" onclick="clearQueue()">Clear All</button>
                </div>
            </div>
        </div>

        <!-- Split Compare Mode -->
        <div id="splitCompareMode" class="split-compare-container">
            <div class="split-compare-header">
                <span class="split-compare-title">Split Compare Mode</span>
                <span class="split-compare-hint">Write prompts for each side, then generate one at a time</span>
                <button onclick="toggleSplitCompare()" class="compare-exit-btn">Exit</button>
            </div>
            <div class="split-compare-panels">
                <div class="split-panel" id="splitPanelA">
                    <div class="split-panel-header">
                        <span class="split-panel-label">A</span>
                        <span class="split-panel-title">Left Side</span>
                    </div>
                    <textarea id="promptA" class="split-prompt" placeholder="Enter prompt for left image..."></textarea>
                    <button onclick="generateSplit('A')" class="split-generate-btn" id="generateBtnA">Generate A</button>
                    <div class="split-result" id="resultA">
                        <div class="split-placeholder">Result will appear here</div>
                    </div>
                </div>
                <div class="split-divider"></div>
                <div class="split-panel" id="splitPanelB">
                    <div class="split-panel-header">
                        <span class="split-panel-label">B</span>
                        <span class="split-panel-title">Right Side</span>
                    </div>
                    <textarea id="promptB" class="split-prompt" placeholder="Enter prompt for right image..."></textarea>
                    <button onclick="generateSplit('B')" class="split-generate-btn" id="generateBtnB">Generate B</button>
                    <div class="split-result" id="resultB">
                        <div class="split-placeholder">Result will appear here</div>
                    </div>
                </div>
            </div>
        </div>

        <div id="status">
            <div id="statusText"></div>
            <div class="progress-container" id="progressContainer" style="display:none;">
                <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
                <div class="progress-text">
                    <span id="stepText">Step 0/4</span>
                    <span id="percentText">0%</span>
                    <span id="timeRemaining"></span>
                </div>
            </div>
        </div>

        <div id="result"></div>

        <div class="examples">
            <div class="examples-header">
                <h3>Prompt Ideas</h3>
                <button type="button" class="btn-tertiary btn-sm" onclick="showGenerateTemplates()">Templates</button>
            </div>
            <div class="prompt-ideas-grid">
                <span class="example-btn" onclick="setPrompt('A cute robot cat in a cozy coffee shop, warm lighting, digital art')">Robot Cat</span>
                <span class="example-btn" onclick="setPrompt('Hyperrealistic portrait of a woman with freckles, natural lighting, sharp focus, professional photography')">Portrait</span>
                <span class="example-btn" onclick="setPrompt('Japanese garden with cherry blossoms and wooden bridge, watercolor style')">Japanese Garden</span>
                <span class="example-btn" onclick="setPrompt('Cyberpunk city at night with neon signs and rain reflections, cinematic')">Cyberpunk</span>
                <span class="example-btn" onclick="setPrompt('Magical forest with glowing mushrooms and fireflies, fantasy art, detailed')">Magic Forest</span>
                <span class="example-btn" onclick="setPrompt('Majestic dragon flying over mountains at sunset, epic fantasy, highly detailed')">Dragon</span>
                <span class="example-btn" onclick="setPrompt('Cozy cabin in snowy mountains, warm light from windows, winter atmosphere')">Winter Cabin</span>
                <span class="example-btn" onclick="setPrompt('Futuristic spaceship interior, sci-fi, volumetric lighting, ultra detailed')">Spaceship</span>
            </div>
        </div>
    </div>

    <!-- Edit Tab -->
    <div id="tab-edit" class="tab-content">
        <div class="input-section">
            <label>Upload Image to Edit</label>
            <div class="btn-row-fill">
                <button type="button" class="btn-secondary" onclick="document.getElementById('imageUpload').click()">Upload File</button>
                <button type="button" class="btn-secondary" onclick="openGalleryPicker()">From Gallery</button>
            </div>
            <div class="upload-area" id="uploadArea" onclick="document.getElementById('imageUpload').click()">
                <div id="uploadPlaceholder">Click or drag image here</div>
                <img id="uploadPreview" class="upload-preview hidden">
                <input type="file" id="imageUpload" accept="image/*" class="hidden" onchange="handleUpload(event)">
            </div>

            <label for="editPrompt">What changes do you want?</label>
            <textarea id="editPrompt" placeholder="e.g., Change the sky to sunset, Add a rainbow, Make it look like winter"></textarea>

            <!-- Edit Mode Selection - Simple Row of Buttons -->
            <div class="edit-mode-section">
                <label>Edit Mode & Tools</label>
                <div class="edit-mode-buttons">
                    <button type="button" id="modeStandard" class="mode-btn active" onclick="selectEditMode('standard')">Standard</button>
                    <button type="button" id="modeAngles" class="mode-btn" onclick="selectEditMode('angles')">Angles</button>
                    <button type="button" id="modeUpscale" class="mode-btn" onclick="selectEditMode('upscale')">Upscale</button>
                    <span class="btn-divider"></span>
                    <button type="button" class="btn-tertiary" onclick="showEditCheatsheet()">Templates</button>
                    <button type="button" id="angleCheatBtn" class="btn-tertiary btn-warning hidden" onclick="showAngleCheatsheet()">Angle Guide</button>
                </div>
                <!-- Upscale resolution options (shown when upscale mode selected) -->
                <div id="upscaleControls" class="upscale-controls">
                    <div class="upscale-options">
                        <button type="button" class="upscale-btn active" data-res="2K" onclick="selectUpscale('2K')">
                            <strong>2K</strong> <span class="upscale-hint">~2048px</span>
                        </button>
                        <button type="button" class="upscale-btn" data-res="4K" onclick="selectUpscale('4K')">
                            <strong>4K</strong> <span class="upscale-hint">~4096px</span>
                        </button>
                    </div>
                    <input type="hidden" id="upscaleRes" value="2K">
                </div>
                <!-- Hidden inputs for compatibility -->
                <input type="hidden" id="angleDirection" value="front">
                <input type="hidden" id="angleElevation" value="eye">
                <input type="hidden" id="angleDistance" value="medium">
            </div>

            <button onclick="editImage()" id="editBtn">Apply Edit</button>
        </div>
        <div id="editResult"></div>
    </div>

    <!-- Video Tab -->
    <div id="tab-video" class="tab-content">
        <div class="input-section">
            <!-- Video Mode Selection -->
            <label>Video Generation Mode</label>
            <div class="model-selector">
                <button type="button" id="videoModeT2V" class="model-btn active" onclick="selectVideoMode('t2v')">
                    <span class="model-title">Text to Video</span>
                    <span class="model-subtitle">Generate from description</span>
                </button>
                <button type="button" id="videoModeI2V" class="model-btn" onclick="selectVideoMode('i2v')">
                    <span class="model-title">Image to Video</span>
                    <span class="model-subtitle">Animate an image</span>
                </button>
            </div>
            <div id="i2vLimitationHint" class="option-hint" style="display: block; margin-top: -8px; margin-bottom: 12px;">LTX only supports Text to Video. Select Hunyuan or Wan for Image to Video.</div>

            <!-- Video Model Selection -->
            <label>Model</label>
            <div class="model-selector">
                <button type="button" id="videoModelLTX" class="model-btn active" onclick="selectVideoModel('ltx')">
                    <span class="model-title">LTX 2B</span>
                    <span class="model-subtitle">Fast ~30s</span>
                </button>
                <button type="button" id="videoModelHunyuan" class="model-btn" onclick="selectVideoModel('hunyuan')">
                    <span class="model-title">Hunyuan 13B</span>
                    <span class="model-subtitle">Quality ~3m</span>
                </button>
                <button type="button" id="videoModelWan" class="model-btn" onclick="selectVideoModel('wan')">
                    <span class="model-title">Wan 14B</span>
                    <span class="model-subtitle">Best ~5m</span>
                </button>
            </div>

            <!-- Image upload for I2V (hidden by default) -->
            <div id="videoImageUploadSection" class="hidden">
                <label>Start Image</label>
                <div class="btn-row-fill">
                    <button type="button" class="btn-secondary" onclick="document.getElementById('videoImageUpload').click()">Upload File</button>
                    <button type="button" class="btn-secondary" onclick="openVideoGalleryPicker()">From Gallery</button>
                </div>
                <div class="upload-area upload-area-compact" id="videoUploadArea" onclick="document.getElementById('videoImageUpload').click()">
                    <div id="videoUploadPlaceholder">Click or drag image here</div>
                    <img id="videoUploadPreview" class="upload-preview hidden">
                    <input type="file" id="videoImageUpload" accept="image/*" class="hidden" onchange="handleVideoUpload(event)">
                </div>
            </div>

            <label for="videoPrompt">Describe your video</label>
            <div class="textarea-container">
                <textarea id="videoPrompt" placeholder="A majestic eagle soaring through clouds at golden hour, cinematic drone shot..."></textarea>
                <div class="ai-buttons">
                    <button type="button" onclick="refineVideoPrompt('refine')" title="Refine with AI" class="ai-float-btn">AI</button>
                    <button type="button" onclick="refineVideoPrompt('expand')" title="Expand prompt" class="ai-float-btn">+</button>
                </div>
            </div>

            <!-- Video Settings -->
            <div class="options-row-spaced">
                <div class="option-group">
                    <label>Resolution</label>
                    <select id="videoResolution" onchange="updateVideoEstimate()">
                        <option value="480p" selected>480p (832×480)</option>
                        <option value="576p">576p (1024×576)</option>
                        <option value="720p">720p (1280×720)</option>
                    </select>
                </div>
                <div class="option-group">
                    <label>Duration</label>
                    <select id="videoDuration" onchange="updateVideoEstimate()">
                        <option value="41">~2.5 sec (41 frames)</option>
                        <option value="81" selected>~5 sec (81 frames)</option>
                        <option value="121">~7.5 sec (121 frames)</option>
                    </select>
                </div>
                <div class="option-group">
                    <label>Seed</label>
                    <input type="number" id="videoSeed" placeholder="Random">
                </div>
            </div>

            <!-- Advanced Video Options -->
            <div class="advanced-toggle" onclick="toggleVideoAdvanced()">Advanced Options</div>
            <div class="advanced-section" id="videoAdvancedSection">
                <div class="option-group" style="margin-bottom: var(--space-3);">
                    <label for="videoNegativePrompt">Negative Prompt</label>
                    <span class="option-hint">What to avoid in the video</span>
                    <textarea id="videoNegativePrompt" placeholder="blurry, low quality, distorted, watermark, static, jittery" class="textarea-short"></textarea>
                </div>
            </div>

            <div class="time-estimate" id="videoTimeEstimate">Estimated: ~4-5 minutes (480p, 5 sec)</div>

            <div class="btn-row">
                <button id="videoGenerateBtn" onclick="generateVideo()">Generate Video</button>
                <button id="videoCancelBtn" class="btn-cancel" onclick="cancelVideoGeneration()" style="display:none;">Cancel</button>
            </div>

            <!-- Video Progress -->
            <div id="videoStatus" style="display: none;">
                <div id="videoStatusText"></div>
                <div class="progress-container" id="videoProgressContainer">
                    <div class="progress-bar"><div class="progress-fill" id="videoProgressFill"></div></div>
                    <div class="progress-text">
                        <span id="videoStepText">Step 0/30</span>
                        <span id="videoPercentText">0%</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Video Result -->
        <div id="videoResult"></div>

        <!-- Video Prompt Ideas -->
        <div class="examples">
            <h3>Video Prompt Ideas</h3>
            <div class="prompt-ideas-grid">
                <span class="example-btn" onclick="setVideoPrompt('A majestic eagle soaring through clouds at golden hour, cinematic aerial shot')">Eagle Flight</span>
                <span class="example-btn" onclick="setVideoPrompt('Ocean waves crashing on rocky shore at sunset, slow motion, dramatic lighting')">Ocean Waves</span>
                <span class="example-btn" onclick="setVideoPrompt('City street at night with rain reflections and neon signs, cyberpunk atmosphere')">Rainy City</span>
                <span class="example-btn" onclick="setVideoPrompt('Timelapse of flowers blooming in a garden, soft natural lighting')">Flowers Blooming</span>
                <span class="example-btn" onclick="setVideoPrompt('Astronaut floating in space with Earth in background, cinematic')">Space Walk</span>
                <span class="example-btn" onclick="setVideoPrompt('Campfire burning in forest at night, sparks flying, cozy atmosphere')">Campfire</span>
            </div>
        </div>

        <!-- Model Status -->
        <div id="videoModelStatus" class="status-card">
            <h4>Video Models Ready</h4>
            <p>Three video generation models (Q4 GGUF quantized for M-series Macs):</p>
            <div class="model-list">
                <div class="model-list-item">
                    <strong>LTX 2B</strong> - Fastest (~30s)
                    <div class="model-file"><code>ltxv-2b-distilled-Q4_K_M.gguf</code> (1.2GB)</div>
                </div>
                <div class="model-list-item">
                    <strong>Hunyuan 13B</strong> - Quality (~3min)
                    <div class="model-file"><code>hunyuan-video-t2v-Q4_K_M.gguf</code> (7.3GB)</div>
                </div>
                <div class="model-list-item">
                    <strong>Wan 14B</strong> - Best quality (~5min)
                    <div class="model-file"><code>wan2.1-t2v-14b-Q4_K_M.gguf</code> (9.4GB)</div>
                    <div class="model-file"><code>wan2.1-i2v-14b-Q4_K_M.gguf</code> (11GB) - I2V</div>
                </div>
            </div>
        </div>
    </div>

    <!-- Audio Tab -->
    <div id="tab-audio" class="tab-content">
        <div class="input-section">
            <label>Tags (Style &amp; Genre)</label>
            <div class="textarea-container">
                <textarea id="audioTags" placeholder="electronic, upbeat, 120bpm, synth, energetic"></textarea>
                <div class="ai-buttons">
                    <button type="button" onclick="refineAudioPrompt('refine')" title="Refine with AI" class="ai-float-btn">AI</button>
                    <button type="button" onclick="refineAudioPrompt('expand')" title="Expand prompt" class="ai-float-btn">+</button>
                </div>
            </div>

            <label>Lyrics (Optional)</label>
            <div class="option-hint">Use structure tags: [verse], [chorus], [bridge], [intro], [outro]. Prefix lines with language: [en], [zh], [ja], [ko], [es], etc.</div>
            <textarea id="audioLyrics" rows="6" placeholder="[verse]
Walking through the city lights
Stars are shining bright tonight

[chorus]
This is where we belong
Dancing to our favorite song"></textarea>

            <div class="options-row-spaced">
                <div class="option-group">
                    <label>Duration</label>
                    <select id="audioDuration">
                        <option value="30">30 seconds</option>
                        <option value="60" selected>60 seconds</option>
                        <option value="90">90 seconds</option>
                        <option value="120">2 minutes</option>
                        <option value="180">3 minutes</option>
                        <option value="240">4 minutes</option>
                    </select>
                </div>
                <div class="option-group">
                    <label>Format</label>
                    <select id="audioFormat">
                        <option value="flac" selected>FLAC (Lossless)</option>
                        <option value="mp3">MP3 (320k)</option>
                        <option value="opus">Opus (128k)</option>
                    </select>
                </div>
            </div>

            <!-- Advanced Audio Options -->
            <div class="advanced-toggle" onclick="toggleAudioAdvanced()">Advanced Options</div>
            <div class="advanced-section" id="audioAdvancedSection">
                <div class="option-group" style="margin-bottom: var(--space-3);">
                    <label>Lyrics Strength</label>
                    <input type="range" id="audioLyricsStrength" min="0" max="2" step="0.1" value="1.0" oninput="updateLyricsStrengthDisplay()">
                    <span id="lyricsStrengthDisplay" class="range-value">1.0</span>
                </div>
                <div class="option-group" style="margin-bottom: var(--space-3);">
                    <label>Seed</label>
                    <div class="seed-row">
                        <button type="button" class="seed-btn" onclick="randomizeAudioSeed()">Random</button>
                        <input type="number" id="audioSeed" placeholder="Random">
                    </div>
                </div>
            </div>

            <div class="btn-row">
                <button id="audioGenerateBtn" onclick="generateAudio()">Generate Music</button>
                <button id="audioCancelBtn" class="btn-cancel" onclick="cancelAudioGeneration()" style="display:none;">Cancel</button>
            </div>

            <!-- Audio Progress -->
            <div id="audioStatus" style="display: none;">
                <div id="audioStatusText"></div>
                <div class="progress-bar">
                    <div id="audioProgressBar" class="progress-fill" style="width: 0%"></div>
                </div>
            </div>
        </div>

        <!-- Audio Player -->
        <div id="audioResult" class="audio-result-container"></div>

        <!-- Audio Prompt Ideas -->
        <div class="examples">
            <h3>Music Style Ideas</h3>
            <div class="prompt-ideas-grid">
                <span class="example-btn" onclick="setAudioTags('electronic, upbeat, 120bpm, synth, energetic, dance')">Electronic Dance</span>
                <span class="example-btn" onclick="setAudioTags('acoustic, folk, gentle, guitar, peaceful, nature')">Acoustic Folk</span>
                <span class="example-btn" onclick="setAudioTags('jazz, smooth, piano, saxophone, relaxing, lounge')">Smooth Jazz</span>
                <span class="example-btn" onclick="setAudioTags('rock, guitar, drums, energetic, powerful, band')">Rock Band</span>
                <span class="example-btn" onclick="setAudioTags('classical, orchestral, piano, strings, elegant, cinematic')">Classical Orchestra</span>
                <span class="example-btn" onclick="setAudioTags('hip-hop, beats, bass, urban, rhythm, flow')">Hip-Hop Beats</span>
            </div>
        </div>

        <!-- Audio Model Status -->
        <div id="audioModelStatus" class="status-card">
            <h4>Audio Generation Model</h4>
            <p>ACE-Step 3.5B - Music generation with lyrics support</p>
            <div class="model-list">
                <div class="model-list-item">
                    <strong>Model</strong>
                    <div class="model-file"><code>ace_step_v1_3.5b.safetensors</code></div>
                </div>
                <div class="model-list-item">
                    <strong>Features</strong>
                    <div class="model-file">19 languages, lyrics with structure tags, up to 4 minutes</div>
                </div>
            </div>
        </div>
    </div>

    <!-- 3D Tab -->
    <div id="tab-3d" class="tab-content">
        <div class="input-section">
            <label>Source Image</label>
            <div class="option-hint">Upload an image to convert to 3D. Best results with objects on plain backgrounds.</div>
            <div class="btn-row-fill">
                <button type="button" class="btn-secondary" onclick="document.getElementById('3dImageUpload').click()">Upload Image</button>
                <button type="button" class="btn-secondary" onclick="open3DGalleryPicker()">From Gallery</button>
            </div>
            <div class="upload-area" id="3dUploadArea" onclick="document.getElementById('3dImageUpload').click()">
                <div id="3dUploadPlaceholder">Click or drag image here</div>
                <img id="3dUploadPreview" class="upload-preview hidden">
                <input type="file" id="3dImageUpload" accept="image/*" class="hidden" onchange="handle3DUpload(event)">
            </div>

            <div class="options-row-spaced">
                <div class="option-group">
                    <label>Resolution</label>
                    <select id="3dResolution">
                        <option value="128">128 (Fast)</option>
                        <option value="256" selected>256 (Balanced)</option>
                        <option value="512">512 (High Detail)</option>
                    </select>
                </div>
                <div class="option-group">
                    <label>Algorithm</label>
                    <select id="3dAlgorithm">
                        <option value="surface net" selected>Surface Net (Smooth)</option>
                        <option value="basic">Basic (Fast)</option>
                    </select>
                </div>
            </div>

            <!-- Advanced 3D Options -->
            <div class="advanced-toggle" onclick="toggle3DAdvanced()">Advanced Options</div>
            <div class="advanced-section" id="3dAdvancedSection">
                <div class="option-group" style="margin-bottom: var(--space-3);">
                    <label>Mesh Threshold</label>
                    <input type="range" id="3dThreshold" min="0.1" max="0.9" step="0.05" value="0.6" oninput="update3DThresholdDisplay()">
                    <span id="thresholdDisplay" class="range-value">0.6</span>
                </div>
                <div class="option-group" style="margin-bottom: var(--space-3);">
                    <label>Seed</label>
                    <div class="seed-row">
                        <button type="button" class="seed-btn" onclick="randomize3DSeed()">Random</button>
                        <input type="number" id="3dSeed" placeholder="Random">
                    </div>
                </div>
            </div>

            <div class="btn-row">
                <button id="3dGenerateBtn" onclick="generate3D()">Generate 3D Model</button>
                <button id="3dCancelBtn" class="btn-cancel" onclick="cancel3DGeneration()" style="display:none;">Cancel</button>
            </div>

            <!-- 3D Progress -->
            <div id="3dStatus" style="display: none;">
                <div id="3dStatusText"></div>
                <div class="progress-bar">
                    <div id="3dProgressBar" class="progress-fill" style="width: 0%"></div>
                </div>
            </div>
        </div>

        <!-- 3D Model Viewer -->
        <div id="3dResult" class="model-viewer-container"></div>

        <!-- 3D Model Status -->
        <div id="3dModelStatus" class="status-card">
            <h4>3D Generation Model</h4>
            <p>Hunyuan3D v2 - Image to 3D mesh conversion</p>
            <div class="model-list">
                <div class="model-list-item">
                    <strong>Shape Model</strong>
                    <div class="model-file"><code>hunyuan3d-dit-v2-0-mini.safetensors</code></div>
                </div>
                <div class="model-list-item">
                    <strong>Output</strong>
                    <div class="model-file">GLB format (viewable in browser, Blender compatible)</div>
                </div>
            </div>
        </div>
    </div>

    <!-- Gallery Tab -->
    <div id="tab-gallery" class="tab-content">
        <div class="filter-tabs">
            <div class="filter-tab active" onclick="filterGallery('all')">All <span id="galleryCount" class="gallery-count"></span></div>
            <div class="filter-tab" onclick="filterGallery('recent')">Recent</div>
            <div class="filter-tab" onclick="filterGallery('favorites')">Favorites</div>
            <div class="filter-tab" onclick="filterGallery('lightning')">Lightning</div>
            <div class="filter-tab" onclick="filterGallery('normal')">Normal</div>
            <div class="filter-tab" onclick="filterGallery('edit')">Edit</div>
            <div class="filter-tab" onclick="enterCompareMode()">Compare</div>
        </div>
        <div id="compareMode" class="compare-mode-container">
            <div class="compare-header">
                <span class="compare-title">Compare Mode: Select 2 images to compare side-by-side</span>
                <button onclick="exitCompareMode()" class="compare-exit-btn">Exit</button>
            </div>
            <div id="compareSlots" class="compare-slots">
                <div id="compareSlot1" class="compare-slot">
                    Click an image for Slot 1
                </div>
                <div id="compareSlot2" class="compare-slot">
                    Click an image for Slot 2
                </div>
            </div>
        </div>
        <div class="gallery" id="gallery"></div>
    </div>

    <!-- Settings Tab -->
    <div id="tab-settings" class="tab-content">
        <div class="input-section">
            <h2>Settings</h2>

            <!-- Provider Toggle -->
            <div class="settings-card">
                <label class="settings-label">Image Generation</label>
                <div class="settings-item active">
                    <div class="settings-item-content">
                        <div class="settings-item-title">Local Mode (ComfyUI + Qwen)</div>
                        <div class="settings-item-desc">Image generation on your Mac</div>
                    </div>
                </div>
            </div>

            <!-- Local Models Info -->
            <div class="settings-card">
                <label class="settings-label">Local Models</label>
                <div class="settings-models">
                    <div class="settings-model-item">
                        <span class="settings-model-label">Image Gen:</span>
                        <span class="settings-model-value">Qwen-Image-2512 Q6_K + Abliterated Text Encoder Q6_K</span>
                    </div>
                    <div class="settings-model-item">
                        <span class="settings-model-label">Image Edit:</span>
                        <span class="settings-model-value">Qwen-Image-Edit-2511 Q4_K_M (balanced VRAM)</span>
                    </div>
                    <div class="settings-model-item">
                        <span class="settings-model-label">Refinement:</span>
                        <span class="settings-model-value">Qwen 2.5 0.5B via Ollama (~350MB)</span>
                    </div>
                </div>
            </div>

            <!-- Status -->
            <div id="settingsStatus" class="settings-status">
                Settings saved!
            </div>
        </div>
    </div>

    <!-- Image Modal -->
    <div class="modal" id="imageModal" onclick="closeModal()">
        <span class="modal-close">&times;</span>
        <img id="modalImage" src="">
    </div>

    <!-- Gallery Picker Modal for Edit Tab -->
    <div class="modal" id="galleryPickerModal" onclick="closeGalleryPicker(event)">
        <div class="modal-panel" onclick="event.stopPropagation()">
            <div class="modal-header">
                <h3>Select Image to Edit</h3>
                <button class="modal-close-btn" onclick="closeGalleryPicker()">&times;</button>
            </div>
            <div id="galleryPickerGrid" class="gallery gallery-compact"></div>
        </div>
    </div>

    <!-- Angle Cheatsheet Modal -->
    <div class="modal" id="angleCheatsheetModal" onclick="closeAngleCheatsheet(event)">
        <div class="modal-panel modal-panel-wide" onclick="event.stopPropagation()">
            <div class="modal-header">
                <h3>Camera Angle Prompt Cheatsheet</h3>
                <button class="modal-close-btn" onclick="closeAngleCheatsheet()">&times;</button>
            </div>

            <div class="modal-description">
                <p><strong>Format:</strong> <code>&lt;sks&gt; [direction] [elevation] [distance]</code></p>
                <p>Copy the angle prompt and paste it into the "Edit Description" field above.</p>
            </div>

            <div class="modal-grid-3">
                <div class="modal-card">
                    <h4>Direction</h4>
                    <div class="modal-card-content">
                        <div><code>front</code> - Face camera</div>
                        <div><code>left</code> - Turn left</div>
                        <div><code>right</code> - Turn right</div>
                        <div><code>back</code> - Turn away</div>
                        <div><code>front_left</code></div>
                        <div><code>front_right</code></div>
                        <div><code>back_left</code></div>
                        <div><code>back_right</code></div>
                    </div>
                </div>

                <div class="modal-card">
                    <h4>Elevation</h4>
                    <div class="modal-card-content">
                        <div><code>overhead</code> - Bird's eye</div>
                        <div><code>high</code> - Above eye</div>
                        <div><code>eye</code> - Eye level</div>
                        <div><code>low</code> - Below eye</div>
                        <div><code>ground</code> - Ground level</div>
                    </div>
                </div>

                <div class="modal-card">
                    <h4>Distance</h4>
                    <div class="modal-card-content">
                        <div><code>extreme_close</code></div>
                        <div><code>close</code> - Close up</div>
                        <div><code>medium_close</code></div>
                        <div><code>medium</code> - Half body</div>
                        <div><code>medium_full</code></div>
                        <div><code>full</code> - Full body</div>
                        <div><code>wide</code> - With scene</div>
                        <div><code>extreme_wide</code></div>
                    </div>
                </div>
            </div>

            <div class="modal-highlight">
                <h4>Examples (click to copy)</h4>
                <div class="modal-btn-row">
                    <button class="modal-chip" onclick="copyAnglePrompt('&lt;sks&gt; right eye medium')">Right profile</button>
                    <button class="modal-chip" onclick="copyAnglePrompt('&lt;sks&gt; left eye medium')">Left profile</button>
                    <button class="modal-chip" onclick="copyAnglePrompt('&lt;sks&gt; front high close')">High angle close</button>
                    <button class="modal-chip" onclick="copyAnglePrompt('&lt;sks&gt; front low full')">Low full body</button>
                    <button class="modal-chip" onclick="copyAnglePrompt('&lt;sks&gt; back_left eye medium')">Over shoulder</button>
                    <button class="modal-chip" onclick="copyAnglePrompt('&lt;sks&gt; front overhead wide')">Bird's eye wide</button>
                </div>
            </div>

            <div class="modal-actions">
                <button onclick="closeAngleCheatsheet()">Got it!</button>
            </div>
        </div>
    </div>

    <!-- Edit Prompt Template Modal -->
    <div class="modal" id="editCheatsheetModal" onclick="closeEditCheatsheet(event)">
        <div class="modal-panel modal-panel-compact" onclick="event.stopPropagation()">
            <div class="modal-header">
                <h3>Edit Templates</h3>
                <button class="modal-close-btn" onclick="closeEditCheatsheet()">&times;</button>
            </div>

            <p class="modal-description">Select a template and fill in keywords, then click "Use Template"</p>

            <div class="form-group">
                <label class="form-label">Template</label>
                <select id="editTemplateSelect" class="form-input" onchange="updateTemplatePreview()">
                    <optgroup label="Background">
                        <option value="Change the background to {location}">Change background to...</option>
                        <option value="Replace background with {scene}">Replace background with...</option>
                        <option value="Add {weather} weather">Add weather...</option>
                    </optgroup>
                    <optgroup label="Lighting">
                        <option value="Add {type} lighting">Add lighting...</option>
                        <option value="Change lighting to {time} atmosphere">Change to time of day...</option>
                        <option value="Add {color} colored lighting">Add colored lighting...</option>
                    </optgroup>
                    <optgroup label="Style">
                        <option value="Make it look like {style}">Make it look like...</option>
                        <option value="Convert to {art_style} style">Convert to art style...</option>
                        <option value="Apply {effect} effect">Apply effect...</option>
                    </optgroup>
                    <optgroup label="Add Elements">
                        <option value="Add {objects} in the scene">Add objects...</option>
                        <option value="Add falling {particles}">Add falling particles...</option>
                        <option value="Add {effect} effects">Add visual effects...</option>
                    </optgroup>
                    <optgroup label="Portrait">
                        <option value="Change hair color to {color}">Change hair color...</option>
                        <option value="Add {accessory}">Add accessory...</option>
                        <option value="Change outfit to {clothing}">Change outfit...</option>
                        <option value="Change expression to {expression}">Change expression...</option>
                    </optgroup>
                    <optgroup label="Mood">
                        <option value="Make it more {mood}">Change mood to...</option>
                        <option value="Add {atmosphere} atmosphere">Add atmosphere...</option>
                    </optgroup>
                </select>
            </div>

            <div class="form-group">
                <label class="form-label">Your Keyword</label>
                <input type="text" id="editTemplateKeyword" class="form-input" placeholder="e.g., sunset beach, neon pink, anime..."
                       onkeyup="updateTemplatePreview()" onkeypress="if(event.key==='Enter')applyTemplate()">
            </div>

            <div class="form-preview">
                <div class="form-preview-label">Preview</div>
                <div id="templatePreview" class="form-preview-text">Select a template and enter a keyword</div>
            </div>

            <div class="form-group">
                <label class="form-label">Quick Keywords (click to use)</label>
                <div id="quickKeywords" class="modal-btn-row"></div>
            </div>

            <div class="btn-row">
                <button onclick="applyTemplate()">Use Template</button>
                <button class="btn-secondary" onclick="closeEditCheatsheet()">Cancel</button>
            </div>
        </div>
    </div>

    <!-- Generate Templates Modal -->
    <div class="modal" id="generateTemplatesModal" onclick="closeGenerateTemplates(event)">
        <div class="modal-panel modal-panel-compact" onclick="event.stopPropagation()">
            <div class="modal-header">
                <h3>Generate Templates</h3>
                <button class="modal-close-btn" onclick="closeGenerateTemplates()">&times;</button>
            </div>

            <p class="modal-description">Select a template, fill in your subject, and generate!</p>

            <div class="form-group">
                <label class="form-label">Template</label>
                <select id="genTemplateSelect" class="form-input" onchange="updateGenTemplatePreview()">
                    <optgroup label="People">
                        <option value="Hyperrealistic portrait of {subject}, natural lighting, sharp focus, professional photography">Realistic Portrait</option>
                        <option value="Anime style illustration of {subject}, vibrant colors, detailed">Anime Character</option>
                        <option value="Fantasy warrior {subject}, epic armor, dramatic lighting, digital art">Fantasy Warrior</option>
                    </optgroup>
                    <optgroup label="Landscapes">
                        <option value="{subject} landscape at sunset, golden hour, cinematic, breathtaking">Sunset Landscape</option>
                        <option value="Magical {subject} with glowing elements, fantasy art, ethereal atmosphere">Fantasy Scene</option>
                        <option value="{subject} in winter with snow, cozy atmosphere, warm lights">Winter Scene</option>
                    </optgroup>
                    <optgroup label="Art Styles">
                        <option value="{subject}, oil painting style, classical art, museum quality">Oil Painting</option>
                        <option value="{subject}, watercolor style, soft colors, artistic">Watercolor</option>
                        <option value="{subject}, Studio Ghibli style, whimsical, animated">Studio Ghibli</option>
                        <option value="{subject}, cyberpunk aesthetic, neon lights, futuristic">Cyberpunk</option>
                    </optgroup>
                    <optgroup label="Animals & Creatures">
                        <option value="Cute {subject}, adorable, fluffy, heartwarming, detailed fur">Cute Animal</option>
                        <option value="Majestic {subject}, powerful, detailed, nature photography style">Majestic Animal</option>
                        <option value="Mythical {subject} creature, fantasy art, magical, highly detailed">Fantasy Creature</option>
                    </optgroup>
                    <optgroup label="Architecture & Objects">
                        <option value="{subject} interior, cozy atmosphere, warm lighting, detailed">Cozy Interior</option>
                        <option value="Futuristic {subject}, sci-fi design, sleek, volumetric lighting">Sci-Fi Design</option>
                        <option value="Ancient {subject}, mysterious, dramatic lighting, epic scale">Ancient/Epic</option>
                    </optgroup>
                </select>
            </div>

            <div class="form-group">
                <label class="form-label">Your Subject</label>
                <input type="text" id="genTemplateSubject" class="form-input" placeholder="e.g., a woman with red hair, mountain range, dragon..."
                       onkeyup="updateGenTemplatePreview()" onkeypress="if(event.key==='Enter')applyGenTemplate()">
            </div>

            <div class="form-preview">
                <div class="form-preview-label">Preview</div>
                <div id="genTemplatePreview" class="form-preview-text">Select a template and enter your subject</div>
            </div>

            <div class="form-group">
                <label class="form-label">Quick Subjects (click to use)</label>
                <div id="genQuickSubjects" class="modal-btn-row"></div>
            </div>

            <div class="btn-row">
                <button onclick="applyGenTemplate()">Use Template</button>
                <button class="btn-secondary" onclick="closeGenerateTemplates()">Cancel</button>
            </div>
        </div>
    </div>

    <script>
        let currentPromptId = null;
        let progressInterval = null;
        let startTime = null;
        let lastSeed = null;
        let lastPrompt = '';
        let uploadedImageData = null;
        let favorites = JSON.parse(localStorage.getItem('qwen_favorites') || '[]');
        let promptHistory = JSON.parse(localStorage.getItem('qwen_history') || '[]');

        // Connection Status Check
        async function checkConnection() {
            const dot = document.getElementById('statusDot');
            const text = document.getElementById('statusText');
            try {
                const response = await fetch('/health', { timeout: 3000 });
                const data = await response.json();
                if (data.comfyui) {
                    dot.className = 'status-dot connected';
                    text.textContent = 'Connected';
                } else {
                    dot.className = 'status-dot disconnected';
                    text.textContent = 'ComfyUI offline';
                }
            } catch (e) {
                dot.className = 'status-dot disconnected';
                text.textContent = 'Disconnected';
            }
        }
        // Check on load and every 30 seconds
        checkConnection();
        setInterval(checkConnection, 30000);

        // Toast Notification System
        function showToast(title, message = '', type = 'info', duration = 3000) {
            const container = document.getElementById('toastContainer');
            const toast = document.createElement('div');
            toast.className = `toast ${type}`;

            const icons = {
                success: '',
                error: '',
                warning: '',
                info: ''
            };

            toast.innerHTML = `
                <span class="toast-icon">${icons[type] || icons.info}</span>
                <div class="toast-content">
                    <div class="toast-title">${title}</div>
                    ${message ? `<div class="toast-message">${message}</div>` : ''}
                </div>
            `;

            container.appendChild(toast);

            // Auto-remove after duration
            setTimeout(() => {
                toast.classList.add('hiding');
                setTimeout(() => toast.remove(), 300);
            }, duration);

            return toast;
        }

        // Generation Queue
        let generationQueue = [];
        let queueProcessing = false;

        function addToQueue() {
            const prompt = document.getElementById('prompt').value.trim();
            if (!prompt) { showToast('Missing Prompt', 'Please enter a prompt first', 'warning'); return; }

            const mode = document.getElementById('mode').value;
            const resolution = parseInt(document.getElementById('resolution').value);
            const aspect = document.getElementById('aspect').value;
            const negativePrompt = document.getElementById('negativePrompt').value.trim();
            const seedInput = document.getElementById('seedInput').value;
            const seed = seedInput ? parseInt(seedInput) : null;
            const sampler = document.getElementById('sampler').value;
            const scheduler = document.getElementById('scheduler').value;

            generationQueue.push({
                id: Date.now(),
                prompt,
                mode,
                resolution,
                aspect,
                negativePrompt,
                seed,
                sampler,
                scheduler,
                model: currentImageModel,
                status: 'pending'
            });

            updateQueueUI();
            document.getElementById('prompt').value = '';
            showToast('Added to Queue', `${generationQueue.length} item(s) in queue`, 'success');
        }

        function updateQueueUI() {
            const container = document.getElementById('queueContainer');
            const list = document.getElementById('queueList');
            const count = document.getElementById('queueCount');
            const startBtn = document.getElementById('queueStartBtn');

            if (generationQueue.length === 0) {
                container.classList.remove('active');
                return;
            }

            container.classList.add('active');
            count.textContent = generationQueue.length;
            startBtn.disabled = queueProcessing || generationQueue.length === 0;
            startBtn.textContent = queueProcessing ? 'Processing...' : 'Start Queue';

            list.innerHTML = generationQueue.map((item, idx) => {
                const statusIcon = item.status === 'processing' ? '...' : item.status === 'done' ? 'Done' : item.status === 'error' ? 'Err' : 'Wait';
                const modeIcon = item.mode === 'lightning' ? 'L' : 'Q';
                return '<div class="queue-item' + (item.status === 'processing' ? ' processing' : '') + '">' +
                    '<span class="queue-item-status">' + statusIcon + '</span>' +
                    '<span class="queue-item-prompt" title="' + item.prompt.replace(/"/g, '&quot;') + '">' + item.prompt + '</span>' +
                    '<span class="queue-item-settings">' + modeIcon + ' ' + item.resolution + 'px</span>' +
                    (item.status === 'pending' ? '<span class="queue-item-remove" onclick="removeFromQueue(' + item.id + ')">x</span>' : '') +
                    '</div>';
            }).join('');
        }

        function removeFromQueue(id) {
            generationQueue = generationQueue.filter(item => item.id !== id);
            updateQueueUI();
        }

        function clearQueue() {
            if (queueProcessing) {
                if (!confirm('Queue is processing. Stop and clear all?')) return;
                queueProcessing = false;
            }
            generationQueue = [];
            updateQueueUI();
        }

        async function processQueue() {
            if (queueProcessing || generationQueue.length === 0) return;

            queueProcessing = true;
            updateQueueUI();

            const btn = document.getElementById('generateBtn');
            const status = document.getElementById('status');
            const statusText = document.getElementById('statusText');
            const result = document.getElementById('result');

            for (let i = 0; i < generationQueue.length; i++) {
                if (!queueProcessing) break;

                const item = generationQueue[i];
                if (item.status !== 'pending') continue;

                item.status = 'processing';
                updateQueueUI();

                btn.disabled = true;
                btn.textContent = 'Queue ' + (i + 1) + '/' + generationQueue.length;
                status.style.display = 'block';
                status.className = 'generating';
                statusText.textContent = 'Processing queue item ' + (i + 1) + '/' + generationQueue.length;

                try {
                    const queueResponse = await fetch('/queue', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            prompt: item.prompt,
                            mode: item.mode,
                            resolution: item.resolution,
                            aspect: item.aspect,
                            negativePrompt: item.negativePrompt,
                            seed: item.seed,
                            sampler: item.sampler,
                            scheduler: item.scheduler,
                            model: item.model || 'qwen'
                        })
                    });
                    const queueData = await queueResponse.json();
                    if (!queueData.prompt_id) throw new Error(queueData.error || 'Failed to queue');

                    currentPromptId = queueData.prompt_id;
                    progressInterval = setInterval(pollProgress, 500);

                    const response = await fetch('/wait?prompt_id=' + currentPromptId);
                    const data = await response.json();

                    clearInterval(progressInterval);

                    if (data.success) {
                        item.status = 'done';
                        item.result = data.image;
                        const filename = data.image.split('/').pop();
                        result.innerHTML = '<img src="' + data.image + '?t=' + Date.now() + '">' +
                            '<div class="result-actions">' +
                            '<a href="' + data.image + '" download="' + filename + '"><button>Download</button></a>' +
                            '<button onclick="toggleFavorite(\\'' + filename + '\\')">Favorite</button>' +
                            '</div>' +
                            '<div class="result-info">Queue item ' + (i + 1) + ' of ' + generationQueue.length + '</div>';
                    } else {
                        item.status = 'error';
                        item.error = data.error;
                    }
                } catch (e) {
                    clearInterval(progressInterval);
                    item.status = 'error';
                    item.error = e.message;
                }

                updateQueueUI();
            }

            queueProcessing = false;
            btn.disabled = false;
            btn.textContent = 'Generate';
            status.className = 'success';
            statusText.textContent = 'Queue complete!';
            updateQueueUI();

            // Remove completed items after a delay
            setTimeout(() => {
                generationQueue = generationQueue.filter(item => item.status === 'pending');
                updateQueueUI();
            }, 3000);
        }

        // Presets configuration
        const PRESETS = {
            quick: { mode: 'lightning', resolution: 512, aspect: 'square' },
            portrait: { mode: 'lightning', resolution: 768, aspect: 'portrait' },
            landscape: { mode: 'lightning', resolution: 768, aspect: 'landscape' },
            wallpaper: { mode: 'lightning', resolution: 1024, aspect: 'landscape' },
            quality: { mode: 'normal', resolution: 768, aspect: 'square' },
            hd_quality: { mode: 'normal', resolution: 1024, aspect: 'square' }
        };

        function showTab(tab) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById('tab-' + tab).classList.add('active');
            if (tab === 'gallery') loadGallery();
        }

        function toggleAdvanced() {
            const section = document.getElementById('advancedSection');
            const toggle = document.querySelector('.advanced-toggle');
            section.classList.toggle('show');
            toggle.textContent = section.classList.contains('show') ? 'Hide Options ▲' : 'Advanced Options ▼';
        }

        function setPrompt(text) {
            document.getElementById('prompt').value = text;
        }

        // Preset functions
        function applyPreset(preset) {
            const p = PRESETS[preset];
            document.getElementById('mode').value = p.mode;
            document.getElementById('resolution').value = p.resolution;
            document.getElementById('aspect').value = p.aspect;
            updateEstimate();

            // Highlight active preset
            document.querySelectorAll('.preset-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelector('[data-preset="' + preset + '"]').classList.add('active');
        }

        function clearPresetHighlight() {
            document.querySelectorAll('.preset-btn').forEach(btn => btn.classList.remove('active'));
        }

        // History functions
        function toggleHistory() {
            const dropdown = document.getElementById('historyDropdown');
            dropdown.classList.toggle('show');
            if (dropdown.classList.contains('show')) {
                historySearchTerm = '';
                renderHistory();
                // Focus search input when opened
                setTimeout(() => {
                    const input = document.getElementById('historySearchInput');
                    if (input) input.focus();
                }, 50);
            }
        }

        // AI-Powered Prompt Refinement - Local (Ollama/Qwen) and Cloud (Gemini)
        let isRefining = false;

        async function refinePromptAI(mode, provider) {
            if (isRefining) return;

            const promptEl = document.getElementById('prompt');
            const prompt = promptEl.value.trim();
            if (!prompt) {
                showToast('Missing Prompt', 'Enter a prompt first', 'warning');
                return;
            }

            isRefining = true;
            const buttons = document.querySelectorAll('.refine-btn');
            buttons.forEach(btn => {
                btn.disabled = true;
                btn.style.opacity = '0.5';
            });
            promptEl.style.opacity = '0.7';
            const providerName = provider === 'ollama' ? 'Local AI' : 'Cloud AI';
            promptEl.placeholder = providerName + ' is enhancing your prompt...';

            try {
                const response = await fetch('/refine', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ prompt: prompt, mode: mode, provider: provider })
                });
                const data = await response.json();

                if (data.success) {
                    promptEl.value = data.refined;
                    showToast('Prompt Enhanced', 'Your prompt has been refined', 'success');
                } else {
                    showToast('Refinement Failed', data.error || 'Unknown error', 'error');
                }
            } catch (e) {
                showToast('Error', e.message, 'error');
            }

            isRefining = false;
            buttons.forEach(btn => {
                btn.disabled = false;
                btn.style.opacity = '1';
            });
            promptEl.style.opacity = '1';
            promptEl.placeholder = 'Describe your image... e.g., A majestic dragon flying over mountains at sunset';
        }

        // Local refinement (Qwen abliterated - uncensored)
        function refineLocal(mode) { refinePromptAI(mode, 'ollama'); }

        let historySearchTerm = '';

        function renderHistory(searchTerm = '') {
            const dropdown = document.getElementById('historyDropdown');
            document.getElementById('historyCount').textContent = '(' + promptHistory.length + ')';

            // Filter history based on search
            const filtered = searchTerm
                ? promptHistory.filter(item =>
                    item.prompt.toLowerCase().includes(searchTerm.toLowerCase()) ||
                    item.mode.toLowerCase().includes(searchTerm.toLowerCase()))
                : promptHistory;

            let html = '<div class="history-search"><input type="text" id="historySearchInput" placeholder="Search prompts..." value="' + (searchTerm || '') + '" oninput="filterHistory(this.value)"></div>';

            if (filtered.length === 0) {
                html += '<div class="history-empty">' + (searchTerm ? 'No matching prompts' : 'No recent prompts') + '</div>';
            } else {
                html += '<div class="history-list">' + filtered.slice(0, 20).map((item, i) => {
                    const originalIndex = promptHistory.indexOf(item);
                    return '<div class="history-item">' +
                        '<div class="history-item-content">' +
                        '<div class="history-item-text" onclick="useHistoryPrompt(' + originalIndex + ')">' +
                        '<div>' + item.prompt.substring(0, 60) + (item.prompt.length > 60 ? '...' : '') + '</div>' +
                        '<div class="history-meta">' + item.mode + ' | ' + item.resolution + 'px' +
                        (item.timestamp ? ' | ' + new Date(item.timestamp).toLocaleDateString() : '') + '</div>' +
                        '</div>' +
                        '<span class="history-item-delete" onclick="event.stopPropagation(); deleteHistoryItem(' + originalIndex + ')" title="Delete">×</span>' +
                        '</div></div>';
                }).join('') + '</div>';
            }

            dropdown.innerHTML = html;
        }

        function filterHistory(term) {
            historySearchTerm = term;
            renderHistory(term);
            // Keep focus on search input
            setTimeout(() => {
                const input = document.getElementById('historySearchInput');
                if (input) {
                    input.focus();
                    input.setSelectionRange(input.value.length, input.value.length);
                }
            }, 0);
        }

        async function deleteHistoryItem(index) {
            try {
                const response = await fetch('/delete-history', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({index: index})
                });
                const data = await response.json();
                if (data.success) {
                    promptHistory = data.history;
                    renderHistory();
                }
            } catch (e) {
                console.log('Failed to delete history item');
            }
        }

        function useHistoryPrompt(index) {
            const item = promptHistory[index];
            document.getElementById('prompt').value = item.prompt;
            document.getElementById('mode').value = item.mode;
            document.getElementById('resolution').value = item.resolution;
            document.getElementById('aspect').value = item.aspect;
            if (item.negativePrompt) document.getElementById('negativePrompt').value = item.negativePrompt;
            updateEstimate();
            document.getElementById('historyDropdown').classList.remove('show');
        }

        function addToHistory(prompt, mode, resolution, aspect, negativePrompt) {
            const item = { prompt, mode, resolution, aspect, negativePrompt, timestamp: Date.now() };
            // Remove duplicate if exists
            promptHistory = promptHistory.filter(h => h.prompt !== prompt);
            promptHistory.unshift(item);
            promptHistory = promptHistory.slice(0, 20); // Keep last 20
            localStorage.setItem('qwen_history', JSON.stringify(promptHistory));
            fetch('/history', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(item) });
        }

        function updateEstimate() {
            const mode = document.getElementById('mode').value;
            const resolution = parseInt(document.getElementById('resolution').value);
            const batch = parseInt(document.getElementById('batchSize').value) || 1;

            let totalTime;
            if (currentImageModel === 'zimage') {
                // Z-Image Turbo: 8 steps, very fast (~30-60 seconds based on resolution)
                const zimageTime = { 512: 0.5, 768: 0.75, 1024: 1, 1536: 2 };
                totalTime = (zimageTime[resolution] || 1) * batch;
            } else {
                // Qwen times
                const times = { lightning: { 512: 1, 768: 2, 1024: 3, 1536: 5 }, normal: { 512: 7, 768: 12, 1024: 16, 1536: 25 } };
                totalTime = (times[mode][resolution] || times[mode][1024]) * batch;
            }
            // Standardized time format: ~30s, ~1m, ~3m
            let timeStr;
            if (totalTime < 1) {
                timeStr = '~' + Math.round(totalTime * 60) + 's';
            } else {
                timeStr = '~' + Math.round(totalTime) + 'm';
            }
            const batchStr = batch > 1 ? ' (' + batch + ' images)' : '';
            document.getElementById('timeEstimate').textContent = 'Estimated: ' + timeStr + batchStr;
        }

        function formatTime(seconds) {
            if (seconds < 60) return Math.round(seconds) + 's';
            return Math.floor(seconds / 60) + 'm ' + Math.round(seconds % 60) + 's';
        }

        async function pollProgress() {
            if (!currentPromptId) return;
            try {
                const response = await fetch('/progress?prompt_id=' + currentPromptId);
                const data = await response.json();
                const progressFill = document.getElementById('progressFill');
                const stepText = document.getElementById('stepText');
                const timeRemaining = document.getElementById('timeRemaining');
                const progressContainer = document.getElementById('progressContainer');
                const statusText = document.getElementById('statusText');

                const percentText = document.getElementById('percentText');
                if (data.status === 'loading') {
                    statusText.textContent = data.message;
                    progressContainer.style.display = 'none';
                } else if (data.status === 'generating') {
                    progressContainer.style.display = 'block';
                    const percent = Math.round((data.current_step / data.total_steps) * 100);
                    progressFill.style.width = percent + '%';
                    stepText.textContent = 'Step ' + data.current_step + '/' + data.total_steps;
                    percentText.textContent = percent + '%';
                    statusText.textContent = 'Generating...';
                    if (data.current_step > 0 && startTime) {
                        const elapsed = (Date.now() - startTime) / 1000;
                        const remaining = (elapsed / data.current_step) * (data.total_steps - data.current_step);
                        timeRemaining.textContent = '~' + formatTime(remaining) + ' left';
                    }
                } else if (data.status === 'done') {
                    progressFill.style.width = '100%';
                    percentText.textContent = '100%';
                    timeRemaining.textContent = 'Done!';
                }
            } catch (e) {}
        }

        let currentBatchIndex = 0;
        let totalBatchSize = 1;

        async function pollBatchProgress() {
            if (!currentPromptId) return;
            try {
                const response = await fetch('/progress?prompt_id=' + currentPromptId);
                const data = await response.json();
                const progressFill = document.getElementById('progressFill');
                const stepText = document.getElementById('stepText');
                const timeRemaining = document.getElementById('timeRemaining');
                const progressContainer = document.getElementById('progressContainer');
                const statusText = document.getElementById('statusText');

                // Update batch progress indicator if batch > 1
                if (totalBatchSize > 1) {
                    const batchItem = document.getElementById('batch-item-' + currentBatchIndex);
                    if (batchItem) {
                        const miniProgress = batchItem.querySelector('.mini-progress-fill');
                        if (data.status === 'generating' && miniProgress) {
                            const percent = (data.current_step / data.total_steps) * 100;
                            miniProgress.style.width = percent + '%';
                        }
                    }
                }

                const percentText = document.getElementById('percentText');
                if (data.status === 'loading') {
                    statusText.textContent = totalBatchSize > 1
                        ? 'Image ' + (currentBatchIndex + 1) + '/' + totalBatchSize + ': Loading models...'
                        : data.message;
                    progressContainer.style.display = 'none';
                } else if (data.status === 'generating') {
                    progressContainer.style.display = 'block';
                    const percent = Math.round((data.current_step / data.total_steps) * 100);
                    progressFill.style.width = percent + '%';
                    stepText.textContent = 'Step ' + data.current_step + '/' + data.total_steps;
                    percentText.textContent = percent + '%';
                    statusText.textContent = totalBatchSize > 1
                        ? 'Image ' + (currentBatchIndex + 1) + '/' + totalBatchSize + ' generating...'
                        : 'Generating...';
                    if (data.current_step > 0 && startTime) {
                        const elapsed = (Date.now() - startTime) / 1000;
                        const remaining = (elapsed / data.current_step) * (data.total_steps - data.current_step);
                        timeRemaining.textContent = '~' + formatTime(remaining) + ' left';
                    }
                } else if (data.status === 'done') {
                    progressFill.style.width = '100%';
                    percentText.textContent = '100%';
                    timeRemaining.textContent = 'Done!';
                }
            } catch (e) {}
        }

        function renderBatchProgress(batchSize, currentIndex, completedIndices) {
            let html = '<div class="batch-progress">';
            for (let i = 0; i < batchSize; i++) {
                let className = 'batch-progress-item';
                let icon = '';
                if (completedIndices.includes(i)) {
                    className += ' done';
                    icon = '';
                } else if (i === currentIndex) {
                    className += ' active';
                    icon = '';
                }
                html += '<div id="batch-item-' + i + '" class="' + className + '">' +
                    icon + ' #' + (i + 1);
                if (i === currentIndex && !completedIndices.includes(i)) {
                    html += '<div class="mini-progress"><div class="mini-progress-fill" style="width: 0%"></div></div>';
                }
                html += '</div>';
            }
            html += '</div>';
            return html;
        }

        async function generate() {
            const prompt = document.getElementById('prompt').value.trim();
            if (!prompt) { showToast('Missing Prompt', 'Please enter a prompt first', 'warning'); return; }

            const mode = document.getElementById('mode').value;
            const resolution = parseInt(document.getElementById('resolution').value);
            const aspect = document.getElementById('aspect').value;
            const negativePrompt = document.getElementById('negativePrompt').value.trim();
            const seedInput = document.getElementById('seedInput').value;
            const baseSeed = seedInput ? parseInt(seedInput) : null;
            const sampler = document.getElementById('sampler').value;
            const scheduler = document.getElementById('scheduler').value;
            const batchSize = parseInt(document.getElementById('batchSize').value) || 1;

            lastPrompt = prompt;
            addToHistory(prompt, mode, resolution, aspect, negativePrompt);

            const btn = document.getElementById('generateBtn');
            const status = document.getElementById('status');
            const statusText = document.getElementById('statusText');
            const result = document.getElementById('result');
            const progressContainer = document.getElementById('progressContainer');

            btn.disabled = true;
            document.getElementById('cancelBtn').style.display = 'inline-flex';
            status.style.display = 'block';
            status.className = 'generating';
            progressContainer.style.display = 'none';
            document.getElementById('progressFill').style.width = '0%';
            startTime = Date.now();

            // Initialize batch tracking
            totalBatchSize = batchSize;
            currentBatchIndex = 0;
            const generatedImages = [];
            const seeds = [];
            const completedIndices = [];

            // Show batch progress UI if batch > 1
            if (batchSize > 1) {
                result.innerHTML = renderBatchProgress(batchSize, 0, []);
            } else {
                result.innerHTML = '';
            }

            try {
                for (let i = 0; i < batchSize; i++) {
                    currentBatchIndex = i;
                    const seed = baseSeed ? baseSeed + i : null;
                    btn.textContent = batchSize > 1 ? (i + 1) + '/' + batchSize + '...' : 'Generating...';
                    statusText.textContent = batchSize > 1 ? 'Starting image ' + (i + 1) + '...' : 'Starting...';

                    // Update batch progress UI
                    if (batchSize > 1) {
                        result.innerHTML = renderBatchProgress(batchSize, i, completedIndices);
                    }

                    const queueResponse = await fetch('/queue', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({prompt, mode, resolution, aspect, negativePrompt, seed, sampler, scheduler, model: currentImageModel})
                    });
                    const queueData = await queueResponse.json();
                    if (!queueData.prompt_id) throw new Error(queueData.error || 'Failed to queue');

                    currentPromptId = queueData.prompt_id;
                    seeds.push(queueData.seed);
                    startTime = Date.now(); // Reset start time for each image
                    progressInterval = setInterval(pollBatchProgress, 500);

                    const response = await fetch('/wait?prompt_id=' + currentPromptId);
                    const data = await response.json();

                    clearInterval(progressInterval);

                    if (data.success) {
                        generatedImages.push(data.image);
                        completedIndices.push(i);
                        // Update batch progress to show this one as done
                        if (batchSize > 1 && i < batchSize - 1) {
                            result.innerHTML = renderBatchProgress(batchSize, i + 1, completedIndices);
                        }
                    } else {
                        throw new Error(data.error || 'Generation failed');
                    }
                }

                currentPromptId = null;
                lastSeed = seeds[seeds.length - 1];

                status.className = 'success';
                statusText.textContent = batchSize > 1 ? batchSize + ' images done!' : 'Done!';
                progressContainer.style.display = 'none';

                if (batchSize === 1) {
                    const filename = generatedImages[0].split('/').pop();
                    result.innerHTML = '<img src="' + generatedImages[0] + '?t=' + Date.now() + '">' +
                        '<div class="result-actions">' +
                        '<a href="' + generatedImages[0] + '" download="' + filename + '"><button>Download</button></a>' +
                        '<button onclick="toggleFavorite(\\'' + filename + '\\')">Favorite</button>' +
                        '<button class="btn-secondary" onclick="copySeed()">Copy Seed</button>' +
                        '</div>' +
                        '<div class="result-info">Seed: <span class="seed-display" onclick="copySeed()" title="Click to copy">' + lastSeed + '</span></div>';
                } else {
                    // Grid layout for batch results
                    let html = '<div class="batch-results">';
                    generatedImages.forEach((img, idx) => {
                        const filename = img.split('/').pop();
                        html += '<div class="batch-item">' +
                            '<img src="' + img + '?t=' + Date.now() + '">' +
                            '<div class="batch-item-actions">' +
                            '<a href="' + img + '" download="' + filename + '"><button class="btn-sm">DL</button></a>' +
                            '<button class="btn-sm" onclick="toggleFavorite(\\'' + filename + '\\')">Fav</button>' +
                            '</div>' +
                            '<div class="batch-seed">Seed: ' + seeds[idx] + '</div>' +
                            '</div>';
                    });
                    html += '</div>';
                    result.innerHTML = html;
                }

                document.getElementById('regenerateBtn').disabled = false;
                showToast(batchSize > 1 ? 'Batch Complete' : 'Image Generated', batchSize > 1 ? batchSize + ' images ready' : 'Your image is ready', 'success');
            } catch (e) {
                clearInterval(progressInterval);
                status.className = 'error';
                statusText.textContent = 'Error: ' + e.message;
                showToast('Error', e.message, 'error');
            } finally {
                currentPromptId = null;
                totalBatchSize = 1;
                currentBatchIndex = 0;
                btn.disabled = false;
                btn.textContent = 'Generate';
                document.getElementById('cancelBtn').style.display = 'none';
            }
        }

        function regenerate() {
            document.getElementById('seedInput').value = '';
            generate();
        }

        async function cancelGeneration() {
            if (!currentPromptId) return;
            try {
                await fetch('/interrupt', { method: 'POST' });
                clearInterval(progressInterval);
                currentPromptId = null;
                document.getElementById('status').className = 'error';
                document.getElementById('statusText').textContent = 'Cancelled';
                document.getElementById('generateBtn').disabled = false;
                document.getElementById('generateBtn').textContent = 'Generate';
                document.getElementById('cancelBtn').style.display = 'none';
            } catch (e) {
                console.error('Cancel failed:', e);
            }
        }

        function copySeed() {
            navigator.clipboard.writeText(lastSeed.toString());
            showToast('Seed Copied', `Seed: ${lastSeed}`, 'success');
        }

        // Split Compare Mode Functions
        let splitCompareActive = false;

        function toggleSplitCompare() {
            const container = document.getElementById('splitCompareMode');
            const inputSection = document.querySelector('#tab-generate .input-section');
            const result = document.getElementById('result');
            const examples = document.querySelector('#tab-generate .examples');

            splitCompareActive = !splitCompareActive;

            if (splitCompareActive) {
                container.style.display = 'block';
                inputSection.style.display = 'none';
                result.style.display = 'none';
                examples.style.display = 'none';
            } else {
                container.style.display = 'none';
                inputSection.style.display = 'block';
                result.style.display = 'block';
                examples.style.display = 'block';
            }
        }

        async function generateSplit(side) {
            const prompt = document.getElementById('prompt' + side).value.trim();
            if (!prompt) { showToast('Missing Prompt', `Please enter a prompt for side ${side}`, 'warning'); return; }

            const mode = document.getElementById('mode').value;
            const resolution = parseInt(document.getElementById('resolution').value);
            const aspect = document.getElementById('aspect').value;
            const negativePrompt = document.getElementById('negativePrompt').value.trim();

            const btn = document.getElementById('generateBtn' + side);
            const resultDiv = document.getElementById('result' + side);

            btn.disabled = true;
            btn.textContent = 'Generating...';
            resultDiv.innerHTML = '<div class="split-placeholder">Generating...</div>';

            try {
                const queueResponse = await fetch('/queue', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({prompt, mode, resolution, aspect, negativePrompt, model: currentImageModel})
                });
                const queueData = await queueResponse.json();
                if (!queueData.prompt_id) throw new Error(queueData.error || 'Failed to queue');

                const response = await fetch('/wait?prompt_id=' + queueData.prompt_id);
                const data = await response.json();

                if (data.success) {
                    resultDiv.innerHTML = '<img src="' + data.image + '?t=' + Date.now() + '">';
                } else {
                    resultDiv.innerHTML = '<div class="split-placeholder result-error">Error: ' + data.error + '</div>';
                }
            } catch (e) {
                resultDiv.innerHTML = '<div class="split-placeholder result-error">Error: ' + e.message + '</div>';
            } finally {
                // Always reset button state
                btn.disabled = false;
                btn.textContent = 'Generate ' + side;
            }
        }

        // Compare function - generates same prompt with Lightning vs Normal
        async function compareGenerate() {
            const prompt = document.getElementById('prompt').value.trim();
            if (!prompt) { showToast('Missing Prompt', 'Please enter a prompt first', 'warning'); return; }

            const resolution = parseInt(document.getElementById('resolution').value);
            const aspect = document.getElementById('aspect').value;
            const negativePrompt = document.getElementById('negativePrompt').value.trim();
            const seed = Math.floor(Math.random() * 999999999); // Same seed for both

            const btn = document.getElementById('compareBtn');
            const status = document.getElementById('status');
            const statusText = document.getElementById('statusText');
            const result = document.getElementById('result');

            btn.disabled = true;
            btn.textContent = 'Comparing...';
            status.style.display = 'block';
            status.className = 'generating';
            statusText.textContent = 'Generating Lightning version...';
            result.innerHTML = '';

            try {
                // Generate Lightning version
                const lightningResponse = await fetch('/queue', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({prompt, mode: 'lightning', resolution, aspect, negativePrompt, seed})
                });
                const lightningQueue = await lightningResponse.json();
                if (!lightningQueue.prompt_id) throw new Error('Failed to queue lightning version');

                const lightningResult = await fetch('/wait?prompt_id=' + lightningQueue.prompt_id);
                const lightningData = await lightningResult.json();

                statusText.textContent = 'Generating Normal version (this takes longer)...';

                // Generate Normal version with same seed
                const normalResponse = await fetch('/queue', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({prompt, mode: 'normal', resolution, aspect, negativePrompt, seed})
                });
                const normalQueue = await normalResponse.json();
                if (!normalQueue.prompt_id) throw new Error('Failed to queue normal version');

                const normalResult = await fetch('/wait?prompt_id=' + normalQueue.prompt_id);
                const normalData = await normalResult.json();

                status.className = 'success';
                statusText.textContent = 'Comparison complete!';

                // Show side-by-side comparison
                result.innerHTML =
                    '<h3 class="comparison-title">Lightning vs Normal (Same Seed: ' + seed + ')</h3>' +
                    '<div class="comparison">' +
                    '<div class="comparison-col">' +
                    '<div class="comparison-label">Lightning (4 steps, ~1 min)</div>' +
                    (lightningData.success ? '<img src="' + lightningData.image + '?t=' + Date.now() + '">' : '<p>Failed</p>') +
                    '</div>' +
                    '<div class="comparison-col">' +
                    '<div class="comparison-label">Normal (30 steps, ~7 min)</div>' +
                    (normalData.success ? '<img src="' + normalData.image + '?t=' + Date.now() + '">' : '<p>Failed</p>') +
                    '</div>' +
                    '</div>';

                addToHistory(prompt, 'compare', resolution, aspect, negativePrompt);
            } catch (e) {
                status.className = 'error';
                statusText.textContent = 'Error: ' + e.message;
            } finally {
                // Always reset button state
                btn.disabled = false;
                btn.textContent = 'Compare';
            }
        }

        function toggleFavorite(filename) {
            const index = favorites.indexOf(filename);
            if (index > -1) {
                favorites.splice(index, 1);
            } else {
                favorites.push(filename);
            }
            localStorage.setItem('qwen_favorites', JSON.stringify(favorites));
            fetch('/favorite', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({filename, favorites})
            });
        }

        // ==========================================
        // IMAGE MODEL SELECTION
        // ==========================================
        let currentImageModel = 'qwen';  // qwen, zimage

        function selectImageModel(model) {
            currentImageModel = model;

            // Update button states using CSS classes
            ['Qwen', 'ZImage'].forEach(m => {
                const btn = document.getElementById('imageModel' + m);
                const isActive = model === m.toLowerCase();
                btn.classList.toggle('active', isActive);
            });

            // Show/hide Qwen-specific options
            const qwenPresets = document.getElementById('qwenPresets');
            const modeSelector = document.getElementById('modeSelector');

            if (model === 'zimage') {
                // Z-Image Turbo: hide presets and mode (it's always fast)
                qwenPresets.style.display = 'none';
                modeSelector.style.display = 'none';
                // Set resolution default to 1024 for Z-Image
                document.getElementById('resolution').value = '1024';
            } else {
                // Qwen: show presets and mode
                qwenPresets.style.display = 'block';
                modeSelector.style.display = 'block';
            }

            updateEstimate();
        }

        // ==========================================
        // VIDEO GENERATION FUNCTIONS
        // ==========================================
        let currentVideoMode = 't2v';
        let currentVideoModel = 'ltx';  // ltx, hunyuan, wan
        let videoUploadedImageData = null;
        let currentVideoPromptId = null;

        function selectVideoMode(mode) {
            currentVideoMode = mode;

            // Update button states using CSS classes
            document.getElementById('videoModeT2V').classList.toggle('active', mode === 't2v');
            document.getElementById('videoModeI2V').classList.toggle('active', mode === 'i2v');

            // Show/hide image upload for I2V mode
            document.getElementById('videoImageUploadSection').style.display = mode === 'i2v' ? 'block' : 'none';

            updateVideoEstimate();
        }

        function selectVideoModel(model) {
            currentVideoModel = model;

            // Update button states using CSS classes
            ['LTX', 'Hunyuan', 'Wan'].forEach(m => {
                const btn = document.getElementById('videoModel' + m);
                const isActive = model === m.toLowerCase();
                btn.classList.toggle('active', isActive);
            });

            // LTX doesn't support I2V mode currently
            if (model === 'ltx' && currentVideoMode === 'i2v') {
                selectVideoMode('t2v');
                showToast('Note', 'LTX model only supports Text-to-Video', 'info');
            }

            // Show/hide I2V option based on model support
            const i2vBtn = document.getElementById('videoModeI2V');
            const i2vHint = document.getElementById('i2vLimitationHint');
            if (model === 'ltx') {
                i2vBtn.style.opacity = '0.4';
                i2vBtn.style.pointerEvents = 'none';
                if (i2vHint) i2vHint.style.display = 'block';
            } else {
                i2vBtn.style.opacity = '1';
                i2vBtn.style.pointerEvents = 'auto';
                if (i2vHint) i2vHint.style.display = 'none';
            }

            updateVideoEstimate();
        }

        function handleVideoUpload(event) {
            const file = event.target.files[0];
            if (!file) return;

            const reader = new FileReader();
            reader.onload = function(e) {
                videoUploadedImageData = e.target.result;
                document.getElementById('videoUploadPreview').src = videoUploadedImageData;
                document.getElementById('videoUploadPreview').style.display = 'block';
                document.getElementById('videoUploadPlaceholder').style.display = 'none';
            };
            reader.readAsDataURL(file);
        }

        function openVideoGalleryPicker() {
            // Reuse the gallery picker from edit mode
            showToast('Gallery Picker', 'Opening gallery...', 'info');
            // For now, just switch to gallery tab
            showTab('gallery');
        }

        function setVideoPrompt(prompt) {
            document.getElementById('videoPrompt').value = prompt;
        }

        function toggleVideoAdvanced() {
            const section = document.getElementById('videoAdvancedSection');
            const toggle = section.previousElementSibling;
            const isVisible = section.style.display !== 'none';
            section.style.display = isVisible ? 'none' : 'block';
            toggle.textContent = isVisible ? 'Hide Options ▲' : 'Advanced Options ▼';
        }

        function updateVideoEstimate() {
            const resolution = document.getElementById('videoResolution').value;
            const duration = parseInt(document.getElementById('videoDuration').value);
            const mode = currentVideoMode;
            const model = currentVideoModel;

            // Base times per model (480p, 81 frames)
            let baseTime;
            let modelName;
            switch (model) {
                case 'ltx':
                    baseTime = 0.5;  // ~30 seconds
                    modelName = 'LTX 2B';
                    break;
                case 'hunyuan':
                    baseTime = 3;    // ~3 minutes
                    modelName = 'Hunyuan 13B';
                    break;
                case 'wan':
                default:
                    baseTime = 5;    // ~5 minutes
                    modelName = 'Wan 14B';
                    break;
            }

            // Adjust for resolution
            if (resolution === '576p') baseTime *= 1.5;
            if (resolution === '720p') baseTime *= 2.5;

            // Adjust for duration
            if (duration === 41) baseTime *= 0.6;
            if (duration === 121) baseTime *= 1.5;

            // I2V is slightly faster (not applicable to LTX)
            if (mode === 'i2v' && model !== 'ltx') baseTime *= 0.9;

            const minTime = Math.floor(baseTime);
            const maxTime = Math.ceil(baseTime * 1.3);

            const durationSec = Math.round(duration / 16); // 16 fps
            // Standardized time format: ~30s, ~3m, ~5m
            let timeStr;
            if (minTime < 1) {
                timeStr = '~30s';
            } else if (minTime === maxTime) {
                timeStr = '~' + minTime + 'm';
            } else {
                timeStr = '~' + minTime + '-' + maxTime + 'm';
            }
            document.getElementById('videoTimeEstimate').textContent =
                modelName + ': ' + timeStr + ' (' + resolution + ', ' + durationSec + 's)';
        }

        async function refineVideoPrompt(mode) {
            const prompt = document.getElementById('videoPrompt').value.trim();
            if (!prompt) {
                showToast('Enter Prompt', 'Please enter a prompt to refine', 'warning');
                return;
            }
            // Reuse the local refine function with video context
            const systemPrompt = mode === 'expand'
                ? 'Expand this video prompt with cinematic details, camera movements, and motion descriptions. Keep it under 100 words.'
                : 'Refine this video prompt for better generation. Add motion, camera details, and visual quality terms.';

            try {
                const response = await fetch('/refine', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt, mode, context: 'video' })
                });
                const result = await response.json();
                if (result.refined) {
                    document.getElementById('videoPrompt').value = result.refined;
                    showToast('Refined!', 'Video prompt enhanced', 'success');
                }
            } catch (e) {
                showToast('Error', 'Could not refine prompt', 'error');
            }
        }

        async function generateVideo() {
            const prompt = document.getElementById('videoPrompt').value.trim();
            if (!prompt) {
                showToast('Missing Prompt', 'Please describe your video', 'warning');
                return;
            }

            if (currentVideoMode === 'i2v' && !videoUploadedImageData) {
                showToast('Missing Image', 'Please upload a start image for Image-to-Video mode', 'warning');
                return;
            }

            const resolution = document.getElementById('videoResolution').value;
            const length = parseInt(document.getElementById('videoDuration').value);
            const seedInput = document.getElementById('videoSeed').value;
            const seed = seedInput ? parseInt(seedInput) : null;
            const negativePrompt = document.getElementById('videoNegativePrompt').value.trim();

            const btn = document.getElementById('videoGenerateBtn');
            const cancelBtn = document.getElementById('videoCancelBtn');
            const status = document.getElementById('videoStatus');
            const statusText = document.getElementById('videoStatusText');
            const result = document.getElementById('videoResult');

            btn.disabled = true;
            cancelBtn.style.display = 'inline-flex';
            status.style.display = 'block';
            statusText.textContent = 'Queuing video generation...';
            result.innerHTML = '';

            try {
                const queueResponse = await fetch('/video-queue', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        prompt,
                        model: currentVideoModel,
                        mode: currentVideoMode,
                        resolution,
                        length,
                        seed,
                        negative_prompt: negativePrompt,
                        start_image: currentVideoMode === 'i2v' ? videoUploadedImageData : null
                    })
                });

                const queueData = await queueResponse.json();
                if (queueData.error) {
                    throw new Error(queueData.error);
                }

                currentVideoPromptId = queueData.prompt_id;
                statusText.textContent = 'Generating video... This may take several minutes.';

                // Poll for progress
                pollVideoProgress(queueData.prompt_id);

                // Wait for completion
                const response = await fetch('/video-wait?prompt_id=' + queueData.prompt_id);
                const data = await response.json();

                if (data.success) {
                    statusText.textContent = 'Video generated!';
                    result.innerHTML = `
                        <div class="result-container">
                            <video class="result-video" controls autoplay loop>
                                <source src="${data.video}" type="video/webm">
                                Your browser does not support video playback.
                            </video>
                            <div class="result-actions">
                                <a href="${data.video}" download class="btn-secondary">Download</a>
                                <button class="btn-secondary" onclick="setVideoPrompt(document.getElementById('videoPrompt').value); showToast('Ready', 'Generate another!', 'info');">Regenerate</button>
                            </div>
                        </div>
                    `;
                    showToast('Video Ready!', 'Your video has been generated', 'success');
                } else {
                    throw new Error(data.error || 'Video generation failed');
                }
            } catch (e) {
                statusText.textContent = 'Error: ' + e.message;
                showToast('Error', e.message, 'error');
            } finally {
                btn.disabled = false;
                cancelBtn.style.display = 'none';
                currentVideoPromptId = null;
            }
        }

        async function pollVideoProgress(promptId) {
            const progressFill = document.getElementById('videoProgressFill');
            const stepText = document.getElementById('videoStepText');
            const percentText = document.getElementById('videoPercentText');

            while (currentVideoPromptId === promptId) {
                try {
                    const response = await fetch('/progress?prompt_id=' + promptId);
                    const data = await response.json();

                    if (data.progress !== undefined) {
                        const percent = Math.round(data.progress * 100);
                        progressFill.style.width = percent + '%';
                        percentText.textContent = percent + '%';
                        if (data.current && data.total) {
                            stepText.textContent = `Step ${data.current}/${data.total}`;
                        }
                    }
                } catch (e) {
                    // Ignore polling errors
                }
                await new Promise(r => setTimeout(r, 1000));
            }
        }

        async function cancelVideoGeneration() {
            if (currentVideoPromptId) {
                try {
                    await fetch('/interrupt', { method: 'POST' });
                    showToast('Cancelled', 'Video generation cancelled', 'info');
                } catch (e) {
                    // Ignore
                }
            }
            currentVideoPromptId = null;
            document.getElementById('videoGenerateBtn').disabled = false;
            document.getElementById('videoCancelBtn').style.display = 'none';
            document.getElementById('videoStatus').style.display = 'none';
        }

        // ==========================================
        // AUDIO GENERATION FUNCTIONS
        // ==========================================
        let currentAudioPromptId = null;

        function setAudioTags(tags) {
            document.getElementById('audioTags').value = tags;
        }

        function toggleAudioAdvanced() {
            const section = document.getElementById('audioAdvancedSection');
            const toggle = section.previousElementSibling;
            section.classList.toggle('show');
            toggle.classList.toggle('active');
        }

        function updateLyricsStrengthDisplay() {
            const value = document.getElementById('audioLyricsStrength').value;
            document.getElementById('lyricsStrengthDisplay').textContent = value;
        }

        function randomizeAudioSeed() {
            document.getElementById('audioSeed').value = Math.floor(Math.random() * 999999999);
        }

        async function refineAudioPrompt(mode) {
            const tags = document.getElementById('audioTags').value;
            if (!tags.trim()) {
                showToast('No Tags', 'Enter some style tags first', 'warning');
                return;
            }
            try {
                showToast('Refining...', 'Enhancing music tags', 'info');
                const response = await fetch('/refine', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt: tags, mode, type: 'audio' })
                });
                const result = await response.json();
                if (result.refined) {
                    document.getElementById('audioTags').value = result.refined;
                    showToast('Refined!', 'Music tags enhanced', 'success');
                }
            } catch (e) {
                showToast('Error', 'Could not refine tags', 'error');
            }
        }

        async function generateAudio() {
            const tags = document.getElementById('audioTags').value.trim();
            if (!tags) {
                showToast('Missing Tags', 'Please enter style/genre tags', 'warning');
                return;
            }

            const lyrics = document.getElementById('audioLyrics').value.trim();
            const duration = parseInt(document.getElementById('audioDuration').value);
            const format = document.getElementById('audioFormat').value;
            const lyricsStrength = parseFloat(document.getElementById('audioLyricsStrength').value);
            const seed = document.getElementById('audioSeed').value || null;

            const btn = document.getElementById('audioGenerateBtn');
            const cancelBtn = document.getElementById('audioCancelBtn');
            const status = document.getElementById('audioStatus');
            const statusText = document.getElementById('audioStatusText');
            const result = document.getElementById('audioResult');

            btn.disabled = true;
            cancelBtn.style.display = 'inline-flex';
            status.style.display = 'block';
            statusText.textContent = 'Queuing audio generation...';
            result.innerHTML = '';

            let promptId = null;
            try {
                const queueResponse = await fetch('/audio-queue', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        tags,
                        lyrics,
                        duration,
                        format,
                        lyrics_strength: lyricsStrength,
                        seed
                    })
                });

                const queueResult = await queueResponse.json();
                if (queueResult.error) {
                    throw new Error(queueResult.error);
                }

                promptId = queueResult.prompt_id;
                currentAudioPromptId = promptId;
                const usedSeed = queueResult.seed;
                statusText.textContent = 'Generating music... (this may take 20-60 seconds)';

                // Poll for progress
                pollAudioProgress(promptId);

                // Wait for completion
                const waitResponse = await fetch('/audio-wait?prompt_id=' + promptId);
                const waitResult = await waitResponse.json();

                if (currentAudioPromptId !== promptId) {
                    return;
                }
                currentAudioPromptId = null;

                if (waitResult.error) {
                    throw new Error(waitResult.error);
                }

                if (waitResult.audio) {
                    const audioFilename = String(waitResult.audio);
                    const safeAudioFilename = encodeURIComponent(audioFilename);

                    // Clear previous content
                    result.innerHTML = '';

                    // Create audio player card
                    const card = document.createElement('div');
                    card.className = 'audio-player-card';

                    const info = document.createElement('div');
                    info.className = 'audio-info';

                    const filenameSpan = document.createElement('span');
                    filenameSpan.className = 'audio-filename';
                    filenameSpan.textContent = audioFilename;

                    const seedSpan = document.createElement('span');
                    seedSpan.className = 'audio-seed';
                    seedSpan.textContent = 'Seed: ' + usedSeed;

                    info.appendChild(filenameSpan);
                    info.appendChild(seedSpan);

                    const audioEl = document.createElement('audio');
                    audioEl.controls = true;
                    audioEl.autoplay = true;

                    const sourceEl = document.createElement('source');
                    sourceEl.src = '/output/' + safeAudioFilename;
                    sourceEl.type = 'audio/' + (format === 'mp3' ? 'mpeg' : format);
                    audioEl.appendChild(sourceEl);

                    const downloadLink = document.createElement('a');
                    downloadLink.href = '/output/' + safeAudioFilename;
                    downloadLink.download = audioFilename;
                    downloadLink.className = 'download-btn';
                    downloadLink.textContent = 'Download';

                    card.appendChild(info);
                    card.appendChild(audioEl);
                    card.appendChild(downloadLink);

                    result.appendChild(card);
                    showToast('Complete!', 'Music generated successfully', 'success');
                }
            } catch (e) {
                showToast('Error', e.message || 'Audio generation failed', 'error');
                statusText.textContent = 'Error: ' + e.message;
            } finally {
                if (promptId && currentAudioPromptId && currentAudioPromptId !== promptId) {
                    return;
                }
                btn.disabled = false;
                cancelBtn.style.display = 'none';
                status.style.display = 'none';
            }
        }

        async function pollAudioProgress(promptId) {
            const progressFill = document.getElementById('audioProgressBar');

            while (currentAudioPromptId === promptId) {
                try {
                    const response = await fetch('/progress?prompt_id=' + promptId);
                    const data = await response.json();

                    if (data.progress !== undefined) {
                        const percent = Math.round(data.progress * 100);
                        progressFill.style.width = percent + '%';
                    }
                } catch (e) {
                    // Ignore polling errors
                }
                await new Promise(r => setTimeout(r, 1000));
            }
        }

        async function cancelAudioGeneration() {
            if (currentAudioPromptId) {
                try {
                    await fetch('/interrupt', { method: 'POST' });
                    showToast('Cancelled', 'Audio generation cancelled', 'info');
                } catch (e) {
                    // Ignore
                }
            }
            currentAudioPromptId = null;
            document.getElementById('audioGenerateBtn').disabled = false;
            document.getElementById('audioCancelBtn').style.display = 'none';
            document.getElementById('audioStatus').style.display = 'none';
        }

        // ==========================================
        // 3D GENERATION FUNCTIONS
        // ==========================================
        let current3DPromptId = null;
        let uploaded3DImageData = null;

        function toggle3DAdvanced() {
            const section = document.getElementById('3dAdvancedSection');
            const toggle = section.previousElementSibling;
            section.classList.toggle('show');
            toggle.classList.toggle('active');
        }

        function update3DThresholdDisplay() {
            const value = document.getElementById('3dThreshold').value;
            document.getElementById('thresholdDisplay').textContent = value;
        }

        function randomize3DSeed() {
            document.getElementById('3dSeed').value = Math.floor(Math.random() * 999999999);
        }

        function handle3DUpload(event) {
            const file = event.target.files[0];
            if (!file) return;

            const reader = new FileReader();
            reader.onload = function(e) {
                uploaded3DImageData = e.target.result;
                const preview = document.getElementById('3dUploadPreview');
                const placeholder = document.getElementById('3dUploadPlaceholder');
                preview.src = e.target.result;
                preview.classList.remove('hidden');
                placeholder.style.display = 'none';
            };
            reader.readAsDataURL(file);
        }

        function open3DGalleryPicker() {
            // Reuse gallery picker modal
            const modal = document.getElementById('galleryPickerModal');
            const grid = document.getElementById('galleryPickerGrid');
            modal.classList.add('active');

            fetch('/gallery').then(r => r.json()).then(images => {
                grid.innerHTML = images.map(item => {
                    const q = String.fromCharCode(39);
                    return '<div class="gallery-item gallery-item-compact" onclick="select3DFromGallery(' + q + item.filename + q + ')">' +
                        '<img src="/output/' + item.filename + '">' +
                        '</div>';
                }).join('');
            });
        }

        function select3DFromGallery(filename) {
            fetch('/output/' + filename)
                .then(r => r.blob())
                .then(blob => {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        uploaded3DImageData = e.target.result;
                        const preview = document.getElementById('3dUploadPreview');
                        const placeholder = document.getElementById('3dUploadPlaceholder');
                        preview.src = e.target.result;
                        preview.classList.remove('hidden');
                        placeholder.style.display = 'none';
                    };
                    reader.readAsDataURL(blob);
                });
            closeGalleryPicker();
        }

        async function generate3D() {
            if (!uploaded3DImageData) {
                showToast('Missing Image', 'Please upload an image first', 'warning');
                return;
            }

            const resolution = parseInt(document.getElementById('3dResolution').value);
            const algorithm = document.getElementById('3dAlgorithm').value;
            const threshold = parseFloat(document.getElementById('3dThreshold').value);
            const seed = document.getElementById('3dSeed').value || null;

            const btn = document.getElementById('3dGenerateBtn');
            const cancelBtn = document.getElementById('3dCancelBtn');
            const status = document.getElementById('3dStatus');
            const statusText = document.getElementById('3dStatusText');
            const result = document.getElementById('3dResult');

            btn.disabled = true;
            cancelBtn.style.display = 'inline-flex';
            status.style.display = 'block';
            statusText.textContent = 'Queuing 3D generation...';
            result.innerHTML = '';

            let promptId = null;
            try {
                const queueResponse = await fetch('/3d-queue', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        image: uploaded3DImageData,
                        resolution,
                        algorithm,
                        threshold,
                        seed
                    })
                });

                const queueResult = await queueResponse.json();
                if (queueResult.error) {
                    throw new Error(queueResult.error);
                }

                promptId = queueResult.prompt_id;
                current3DPromptId = promptId;
                statusText.textContent = 'Generating 3D model... (this may take 1-3 minutes)';

                // Poll for progress
                poll3DProgress(promptId);

                // Wait for completion
                const waitResponse = await fetch('/3d-wait?prompt_id=' + promptId);
                const waitResult = await waitResponse.json();

                if (current3DPromptId !== promptId) {
                    return;
                }
                current3DPromptId = null;

                if (waitResult.error) {
                    throw new Error(waitResult.error);
                }

                if (waitResult.mesh) {
                    const meshFilename = String(waitResult.mesh);
                    const safeMeshFilename = encodeURIComponent(meshFilename);

                    // Clear previous content
                    result.innerHTML = '';

                    // Create model viewer card
                    const card = document.createElement('div');
                    card.className = 'model-viewer-card';

                    const viewer = document.createElement('model-viewer');
                    viewer.src = '/output/' + safeMeshFilename;
                    viewer.alt = 'Generated 3D Model';
                    viewer.setAttribute('camera-controls', '');
                    viewer.setAttribute('auto-rotate', '');
                    viewer.setAttribute('shadow-intensity', '1');
                    viewer.style.width = '100%';
                    viewer.style.height = '400px';
                    viewer.style.background = '#1c1c1e';
                    viewer.style.borderRadius = 'var(--radius-lg)';

                    const actions = document.createElement('div');
                    actions.className = 'model-actions';

                    const downloadLink = document.createElement('a');
                    downloadLink.href = '/output/' + safeMeshFilename;
                    downloadLink.download = meshFilename;
                    downloadLink.className = 'download-btn';
                    downloadLink.textContent = 'Download GLB';

                    actions.appendChild(downloadLink);
                    card.appendChild(viewer);
                    card.appendChild(actions);
                    result.appendChild(card);
                    showToast('Complete!', '3D model generated successfully', 'success');
                }
            } catch (e) {
                showToast('Error', e.message || '3D generation failed', 'error');
                statusText.textContent = 'Error: ' + e.message;
            } finally {
                if (promptId && current3DPromptId && current3DPromptId !== promptId) {
                    return;
                }
                btn.disabled = false;
                cancelBtn.style.display = 'none';
                status.style.display = 'none';
            }
        }

        async function poll3DProgress(promptId) {
            const progressFill = document.getElementById('3dProgressBar');

            while (current3DPromptId === promptId) {
                try {
                    const response = await fetch('/progress?prompt_id=' + promptId);
                    const data = await response.json();

                    if (data.progress !== undefined) {
                        const percent = Math.round(data.progress * 100);
                        progressFill.style.width = percent + '%';
                    }
                } catch (e) {
                    // Ignore polling errors
                }
                await new Promise(r => setTimeout(r, 1000));
            }
        }

        async function cancel3DGeneration() {
            if (current3DPromptId) {
                try {
                    await fetch('/interrupt', { method: 'POST' });
                    showToast('Cancelled', '3D generation cancelled', 'info');
                } catch (e) {
                    // Ignore
                }
            }
            current3DPromptId = null;
            document.getElementById('3dGenerateBtn').disabled = false;
            document.getElementById('3dCancelBtn').style.display = 'none';
            document.getElementById('3dStatus').style.display = 'none';
        }

        // Gallery functions
        let compareMode = false;
        let compareSlot1 = null;
        let compareSlot2 = null;
        let galleryData = [];

        function getImageType(img) {
            if (img.includes('edit')) return 'edit';
            if (img.includes('lightning')) return 'lightning';
            if (img.includes('normal') || img.includes('image')) return 'normal';
            return 'lightning';
        }

        function formatFileSize(bytes) {
            if (bytes < 1024) return bytes + ' B';
            if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
            return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
        }

        function formatRelativeTime(timestamp) {
            const now = Date.now() / 1000;
            const diff = now - timestamp;
            if (diff < 60) return 'Just now';
            if (diff < 3600) return Math.floor(diff / 60) + 'm ago';
            if (diff < 86400) return Math.floor(diff / 3600) + 'h ago';
            if (diff < 604800) return Math.floor(diff / 86400) + 'd ago';
            return new Date(timestamp * 1000).toLocaleDateString();
        }

        async function loadGallery() {
            try {
                const response = await fetch('/gallery');
                galleryData = await response.json();
                const gallery = document.getElementById('gallery');
                const countEl = document.getElementById('galleryCount');
                if (countEl) countEl.textContent = '(' + galleryData.length + ')';

                if (galleryData.length === 0) {
                    gallery.innerHTML = '<div class="gallery-empty">No images yet. Generate some!</div>';
                    return;
                }

                gallery.innerHTML = galleryData.map(item => {
                    const img = item.filename;
                    const type = getImageType(img);
                    const q = String.fromCharCode(39);
                    const timeAgo = formatRelativeTime(item.timestamp);
                    const size = formatFileSize(item.size);
                    const typeBadge = type === 'lightning' ? 'Lightning' : type === 'edit' ? 'Edit' : 'Normal';
                    return '<div class="gallery-item" data-type="' + type + '" data-filename="' + img + '" data-timestamp="' + item.timestamp + '">' +
                        '<img src="/output/' + img + '" onclick="handleGalleryClick(' + q + img + q + ')">' +
                        '<div class="gallery-actions">' +
                        '<span class="favorite-star" onclick="event.stopPropagation(); toggleFavorite(' + q + img + q + ')">' + (favorites.includes(img) ? '★' : '☆') + '</span>' +
                        '<span class="delete-btn" onclick="event.stopPropagation(); deleteImage(' + q + img + q + ')">×</span>' +
                        '</div>' +
                        '<div class="gallery-type-badge">' + typeBadge + '</div>' +
                        '<div class="gallery-info"><div class="gallery-info-text"><span>' + timeAgo + '</span><span>' + size + '</span></div></div>' +
                        '</div>';
                }).join('');
            } catch (e) {
                document.getElementById('gallery').innerHTML = '<div class="gallery-empty">Could not load gallery</div>';
            }
        }

        function handleGalleryClick(img) {
            if (compareMode) {
                addToCompare(img);
            } else {
                openModal('/output/' + img);
            }
        }

        function filterGallery(filter) {
            document.querySelectorAll('.filter-tab').forEach(t => t.classList.remove('active'));
            event.target.classList.add('active');
            const now = Date.now() / 1000;
            const oneDayAgo = now - 86400;
            let visibleCount = 0;

            document.querySelectorAll('.gallery-item').forEach(item => {
                const imgName = item.dataset.filename;
                const timestamp = parseFloat(item.dataset.timestamp);
                let show = false;

                if (filter === 'all') show = true;
                else if (filter === 'recent') show = timestamp > oneDayAgo;
                else if (filter === 'favorites') show = favorites.includes(imgName);
                else show = item.dataset.type === filter;

                item.style.display = show ? 'block' : 'none';
                if (show) visibleCount++;
            });

            // Show empty state if no matches
            const gallery = document.getElementById('gallery');
            const existingEmpty = gallery.querySelector('.gallery-empty');
            if (existingEmpty) existingEmpty.remove();

            if (visibleCount === 0) {
                const emptyMsg = {
                    'recent': 'No images from the last 24 hours',
                    'favorites': 'No favorites yet. Mark some as favorites!',
                    'lightning': 'No Lightning mode images',
                    'normal': 'No Normal mode images',
                    'edit': 'No edited images'
                };
                gallery.insertAdjacentHTML('beforeend', '<div class="gallery-empty">' + (emptyMsg[filter] || 'No images') + '</div>');
            }
        }

        function enterCompareMode() {
            compareMode = true;
            compareSlot1 = null;
            compareSlot2 = null;
            document.getElementById('compareMode').style.display = 'block';
            resetCompareSlot('compareSlot1', 'Click an image for Slot 1');
            resetCompareSlot('compareSlot2', 'Click an image for Slot 2');
            document.querySelectorAll('.filter-tab').forEach(t => t.classList.remove('active'));
            event.target.classList.add('active');
        }

        function exitCompareMode() {
            compareMode = false;
            compareSlot1 = null;
            compareSlot2 = null;
            document.getElementById('compareMode').style.display = 'none';
            document.querySelector('.filter-tab').classList.add('active');
        }

        function resetCompareSlot(slotId, placeholder) {
            const slot = document.getElementById(slotId);
            slot.innerHTML = placeholder;
            slot.classList.remove('filled');
            slot.style.border = '';
        }

        function addToCompare(img) {
            const imgHtml = '<img src="/output/' + img + '">';
            const slot1 = document.getElementById('compareSlot1');
            const slot2 = document.getElementById('compareSlot2');

            if (!compareSlot1) {
                compareSlot1 = img;
                slot1.innerHTML = imgHtml;
                slot1.classList.add('filled');
            } else if (!compareSlot2) {
                compareSlot2 = img;
                slot2.innerHTML = imgHtml;
                slot2.classList.add('filled');
            } else {
                // Replace slot 1, shift slot 2
                compareSlot1 = compareSlot2;
                compareSlot2 = img;
                slot1.innerHTML = slot2.innerHTML;
                slot2.innerHTML = imgHtml;
            }
        }

        async function deleteImage(filename) {
            if (!confirm('Delete ' + filename + '?')) return;
            try {
                const response = await fetch('/delete-image', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({filename: filename})
                });
                const data = await response.json();
                if (data.success) {
                    loadGallery();
                    showToast('Image Deleted', 'The image has been removed', 'success');
                } else {
                    showToast('Delete Failed', data.error, 'error');
                }
            } catch (e) {
                showToast('Error', 'Failed to delete image', 'error');
            }
        }

        function openModal(src) {
            document.getElementById('modalImage').src = src;
            document.getElementById('imageModal').classList.add('active');
        }

        function closeModal() {
            document.getElementById('imageModal').classList.remove('active');
        }

        // Upload/Edit functions
        function handleUpload(event) {
            const file = event.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    uploadedImageData = e.target.result;
                    const preview = document.getElementById('uploadPreview');
                    preview.src = uploadedImageData;
                    preview.style.display = 'block';
                };
                reader.readAsDataURL(file);
            }
        }

        // Drag and drop
        const uploadArea = document.getElementById('uploadArea');
        uploadArea.addEventListener('dragover', (e) => { e.preventDefault(); uploadArea.classList.add('dragover'); });
        uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                document.getElementById('imageUpload').files = e.dataTransfer.files;
                handleUpload({target: {files: [file]}});
            }
        });

        // Gallery Picker for Edit Tab
        async function openGalleryPicker() {
            const modal = document.getElementById('galleryPickerModal');
            const grid = document.getElementById('galleryPickerGrid');

            try {
                const response = await fetch('/gallery');
                const images = await response.json();

                if (images.length === 0) {
                    grid.innerHTML = '<div class="gallery-empty">No images in gallery. Generate some first!</div>';
                } else {
                    grid.innerHTML = images.slice(0, 20).map(item => {
                        const img = item.filename;
                        const q = String.fromCharCode(39);
                        return '<div class="gallery-item" style="cursor:pointer;" onclick="selectGalleryImage(' + q + img + q + ')">' +
                            '<img src="/output/' + img + '">' +
                            '</div>';
                    }).join('');
                }
                modal.style.display = 'flex';
            } catch (e) {
                showToast('Error', 'Could not load gallery', 'error');
            }
        }

        function closeGalleryPicker(event) {
            if (!event || event.target.id === 'galleryPickerModal') {
                document.getElementById('galleryPickerModal').style.display = 'none';
            }
        }

        async function selectGalleryImage(filename) {
            // Load image from gallery and convert to base64
            try {
                const response = await fetch('/output/' + filename);
                const blob = await response.blob();
                const reader = new FileReader();
                reader.onload = function(e) {
                    uploadedImageData = e.target.result;
                    const preview = document.getElementById('uploadPreview');
                    preview.src = uploadedImageData;
                    preview.style.display = 'block';
                    document.getElementById('uploadPlaceholder').style.display = 'none';
                    closeGalleryPicker();
                    showToast('Image Selected', 'Ready for editing', 'success');
                };
                reader.readAsDataURL(blob);
            } catch (e) {
                showToast('Error', 'Could not load image', 'error');
            }
        }

        let currentEditMode = 'standard';

        function selectEditMode(mode) {
            currentEditMode = mode;

            // Update visual selection for mode buttons using CSS classes
            document.querySelectorAll('#tab-edit .mode-btn').forEach(el => {
                el.classList.remove('active');
            });
            const selected = document.getElementById('mode' + mode.charAt(0).toUpperCase() + mode.slice(1));
            if (selected) {
                selected.classList.add('active');
            }

            // Show/hide upscale controls and angle guide button
            document.getElementById('upscaleControls').style.display = mode === 'upscale' ? 'block' : 'none';
            document.getElementById('angleCheatBtn').style.display = mode === 'angles' ? 'inline-block' : 'none';

            // Update placeholder based on mode
            if (mode === 'upscale') {
                document.getElementById('editPrompt').placeholder = 'Optional: describe any additional changes, or leave blank for pure upscale';
            } else if (mode === 'angles') {
                document.getElementById('editPrompt').placeholder = 'Enter angle prompt (e.g., <sks> right eye medium) - click Angle Guide for help';
            } else {
                document.getElementById('editPrompt').placeholder = 'Describe what you want to change... (click Templates for ideas)';
            }
        }

        // Angle cheatsheet modal functions
        function showAngleCheatsheet() {
            document.getElementById('angleCheatsheetModal').style.display = 'flex';
        }

        function closeAngleCheatsheet(event) {
            if (!event || event.target === event.currentTarget) {
                document.getElementById('angleCheatsheetModal').style.display = 'none';
            }
        }

        function copyAnglePrompt(prompt) {
            // Decode HTML entities
            const decoded = prompt.replace(/&lt;/g, '<').replace(/&gt;/g, '>');
            navigator.clipboard.writeText(decoded).then(() => {
                showToast('Copied!', 'Angle prompt copied to clipboard', 'success');
                // Also paste into the edit prompt field
                document.getElementById('editPrompt').value = decoded;
                closeAngleCheatsheet();
            }).catch(err => {
                showToast('Copy Failed', 'Please copy manually', 'warning');
            });
        }

        // Edit template modal functions
        const templateKeywords = {
            'location': ['sunset beach', 'mountain landscape', 'city skyline', 'forest', 'desert', 'ocean', 'space', 'castle'],
            'scene': ['tropical paradise', 'snowy mountains', 'cherry blossom garden', 'futuristic city', 'medieval village', 'underwater'],
            'weather': ['rainy', 'snowy', 'foggy', 'stormy', 'sunny', 'cloudy'],
            'type': ['golden hour', 'dramatic', 'soft', 'neon', 'studio', 'natural', 'cinematic', 'rim'],
            'time': ['sunset', 'sunrise', 'night', 'golden hour', 'blue hour', 'midday'],
            'color': ['neon pink', 'blue', 'purple', 'warm orange', 'cool blue', 'green'],
            'style': ['an oil painting', 'a watercolor', 'a photograph', 'a movie scene', 'a dream', 'a vintage photo'],
            'art_style': ['anime', 'pixar', 'studio ghibli', 'impressionist', 'pop art', 'cyberpunk', 'steampunk'],
            'effect': ['vintage film', 'HDR', 'bokeh', 'motion blur', 'double exposure', 'glitch'],
            'objects': ['butterflies', 'fireflies', 'birds', 'flowers', 'stars', 'lanterns', 'bubbles'],
            'particles': ['cherry blossoms', 'snow', 'leaves', 'sparkles', 'confetti', 'rain', 'petals'],
            'hair_color': ['blonde', 'red', 'blue', 'pink', 'silver', 'black', 'purple'],
            'accessory': ['sunglasses', 'a hat', 'earrings', 'a crown', 'glasses', 'a scarf'],
            'clothing': ['formal suit', 'casual outfit', 'elegant dress', 'leather jacket', 'traditional kimono'],
            'expression': ['smiling', 'serious', 'surprised', 'laughing', 'thoughtful', 'confident'],
            'mood': ['dramatic and moody', 'bright and cheerful', 'dark and mysterious', 'romantic', 'ethereal'],
            'atmosphere': ['dreamy', 'mysterious', 'romantic', 'energetic', 'peaceful', 'dramatic']
        };

        function showEditCheatsheet() {
            document.getElementById('editCheatsheetModal').style.display = 'flex';
            document.getElementById('editTemplateKeyword').value = '';
            updateTemplatePreview();
        }

        function closeEditCheatsheet(event) {
            if (!event || event.target === event.currentTarget) {
                document.getElementById('editCheatsheetModal').style.display = 'none';
            }
        }

        function updateTemplatePreview() {
            const template = document.getElementById('editTemplateSelect').value;
            const keyword = document.getElementById('editTemplateKeyword').value.trim();
            const preview = document.getElementById('templatePreview');
            const quickKeywordsDiv = document.getElementById('quickKeywords');

            // Extract placeholder name from template
            const placeholderMatch = template.match(/\\{(\\w+)\\}/);
            const placeholderName = placeholderMatch ? placeholderMatch[1] : '';

            // Update preview
            if (keyword) {
                preview.textContent = template.replace(/\\{[^}]+\\}/, keyword);
                preview.style.color = 'var(--text-primary)';
            } else {
                preview.textContent = template;
                preview.style.color = 'var(--text-tertiary)';
            }

            // Update quick keywords
            const keywords = templateKeywords[placeholderName] || ['example 1', 'example 2', 'example 3'];
            quickKeywordsDiv.innerHTML = keywords.map(kw =>
                `<button class="cheat-btn" onclick="setTemplateKeyword('${kw}')">${kw}</button>`
            ).join('');
        }

        function setTemplateKeyword(keyword) {
            document.getElementById('editTemplateKeyword').value = keyword;
            updateTemplatePreview();
        }

        function applyTemplate() {
            const template = document.getElementById('editTemplateSelect').value;
            const keyword = document.getElementById('editTemplateKeyword').value.trim();

            if (!keyword) {
                showToast('Missing Keyword', 'Please enter a keyword for the template', 'warning');
                return;
            }

            const finalPrompt = template.replace(/\\{[^}]+\\}/, keyword);
            document.getElementById('editPrompt').value = finalPrompt;
            closeEditCheatsheet();
            showToast('Template Applied', 'Prompt ready - click Apply Edit', 'success');
        }

        // Generate Templates Modal Functions
        const genQuickSubjects = {
            'people': ['a woman with curly red hair', 'an old wizard with long beard', 'a young boy with freckles', 'a cyberpunk girl with neon hair'],
            'landscape': ['mountain range', 'tropical beach', 'enchanted forest', 'desert oasis', 'snowy village'],
            'animal': ['golden retriever puppy', 'majestic lion', 'owl', 'red panda', 'phoenix'],
            'object': ['cozy cabin', 'spaceship interior', 'ancient temple', 'steampunk clock']
        };

        function showGenerateTemplates() {
            document.getElementById('generateTemplatesModal').style.display = 'flex';
            document.getElementById('genTemplateSubject').value = '';
            updateGenTemplatePreview();
        }

        function closeGenerateTemplates(event) {
            if (!event || event.target === event.currentTarget) {
                document.getElementById('generateTemplatesModal').style.display = 'none';
            }
        }

        function updateGenTemplatePreview() {
            const template = document.getElementById('genTemplateSelect').value;
            const subject = document.getElementById('genTemplateSubject').value.trim();
            const preview = document.getElementById('genTemplatePreview');
            const quickSubjectsDiv = document.getElementById('genQuickSubjects');

            // Update preview
            if (subject) {
                preview.textContent = template.replace(/\\{subject\\}/g, subject);
                preview.style.color = 'var(--text-primary)';
            } else {
                preview.textContent = template;
                preview.style.color = 'var(--text-tertiary)';
            }

            // Determine category for quick subjects
            const selectedOption = document.getElementById('genTemplateSelect').selectedOptions[0];
            const optGroup = selectedOption.parentElement.label || '';
            let category = 'people';
            if (optGroup.includes('Landscape')) category = 'landscape';
            else if (optGroup.includes('Animal')) category = 'animal';
            else if (optGroup.includes('Architecture')) category = 'object';

            // Update quick subjects
            const subjects = genQuickSubjects[category] || genQuickSubjects['people'];
            quickSubjectsDiv.innerHTML = subjects.map(subj =>
                `<button class="cheat-btn" onclick="setGenTemplateSubject('${subj}')">${subj}</button>`
            ).join('');
        }

        function setGenTemplateSubject(subject) {
            document.getElementById('genTemplateSubject').value = subject;
            updateGenTemplatePreview();
        }

        function applyGenTemplate() {
            const template = document.getElementById('genTemplateSelect').value;
            const subject = document.getElementById('genTemplateSubject').value.trim();

            if (!subject) {
                showToast('Missing Subject', 'Please enter a subject for the template', 'warning');
                return;
            }

            const finalPrompt = template.replace(/\\{subject\\}/g, subject);
            document.getElementById('prompt').value = finalPrompt;
            closeGenerateTemplates();
            showToast('Template Applied', 'Ready to generate!', 'success');
        }

        // Visual angle picker selection
        function selectAngle(element, type) {
            let group, valueField;
            if (type === 'dir') {
                group = document.querySelectorAll('.angle-btn[data-dir]');
                valueField = 'angleDirection';
            } else if (type === 'elev') {
                group = document.querySelectorAll('.elev-btn');
                valueField = 'angleElevation';
            } else if (type === 'dist') {
                group = document.querySelectorAll('.dist-btn');
                valueField = 'angleDistance';
            }

            // Remove active styling from all in group
            group.forEach(btn => {
                btn.style.background = 'var(--glass-bg-hover)';
                btn.style.border = 'none';
            });

            // Add active styling to clicked
            element.style.background = 'var(--accent-bg-active)';
            element.style.border = '2px solid var(--accent)';

            // Update hidden value
            const value = element.dataset.dir || element.dataset.elev || element.dataset.dist;
            document.getElementById(valueField).value = value;
        }

        // Upscale resolution selection
        function selectUpscale(resolution) {
            // Update button styling using CSS classes
            document.querySelectorAll('.upscale-btn').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.res === resolution);
            });
            document.getElementById('upscaleRes').value = resolution;
        }

        function getAnglePrompt() {
            const direction = document.getElementById('angleDirection').value;
            const elevation = document.getElementById('angleElevation').value;
            const distance = document.getElementById('angleDistance').value;
            return '<sks> ' + direction + ' ' + elevation + ' ' + distance;
        }

        async function editImage() {
            if (!uploadedImageData) { showToast('No Image', 'Please upload an image first', 'warning'); return; }
            let editPrompt = document.getElementById('editPrompt').value.trim();

            const useAnglesLora = currentEditMode === 'angles';
            const useUpscaleLora = currentEditMode === 'upscale';

            // For upscale mode, auto-generate prompt if empty
            if (useUpscaleLora) {
                const upscaleRes = document.getElementById('upscaleRes').value;
                const upscaleTrigger = upscaleRes === '4K' ? '<s2k>' : '<s2k>';  // Both use same trigger
                if (!editPrompt) {
                    editPrompt = upscaleTrigger;  // Pure upscale with trigger
                } else {
                    editPrompt = upscaleTrigger + ' ' + editPrompt;  // Upscale + modifications
                }
            } else if (useAnglesLora) {
                // For angles mode, user must provide the angle prompt manually
                if (!editPrompt) {
                    showToast('Missing Angle Prompt', 'Click the cheatsheet button and copy an angle prompt', 'warning');
                    return;
                }
                // Validate it contains the <sks> trigger
                if (!editPrompt.includes('<sks>')) {
                    showToast('Invalid Angle Prompt', 'Angle prompts must include <sks> trigger (see cheatsheet)', 'warning');
                    return;
                }
            } else if (!editPrompt) {
                showToast('Missing Description', 'Please describe the changes', 'warning');
                return;
            }

            // For angles mode, the editPrompt IS the angle prompt
            const anglePrompt = useAnglesLora ? editPrompt : '';

            const originalImage = uploadedImageData;
            document.getElementById('editBtn').disabled = true;
            document.getElementById('editBtn').textContent = 'Processing...';
            document.getElementById('editResult').innerHTML =
                '<div class="result-spinner">' +
                '<div class="spinner"></div>' +
                '<p>Editing image... This may take 2-5 minutes</p>' +
                '</div>';

            try {
                const response = await fetch('/edit', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        image: uploadedImageData,
                        prompt: editPrompt,
                        useAnglesLora: useAnglesLora,
                        useUpscaleLora: useUpscaleLora,
                        anglePrompt: anglePrompt
                    })
                });
                const data = await response.json();
                if (data.success) {
                    const filename = data.image.split('/').pop();
                    document.getElementById('editResult').innerHTML =
                        '<div class="result-grid">' +
                        '<div class="result-grid-item">' +
                        '<div class="result-grid-label">Before</div>' +
                        '<img src="' + originalImage + '" class="result-grid-image">' +
                        '</div>' +
                        '<div class="result-grid-item">' +
                        '<div class="result-grid-label">After</div>' +
                        '<img src="' + data.image + '?t=' + Date.now() + '" class="result-grid-image result-grid-image-highlight">' +
                        '</div>' +
                        '</div>' +
                        '<div class="result-actions">' +
                        '<a href="' + data.image + '" download="' + filename + '"><button>Download</button></a>' +
                        '<button onclick="toggleFavorite(\\'' + filename + '\\')">Favorite</button>' +
                        '</div>';
                    showToast('Edit Complete', 'Your image has been edited', 'success');
                } else {
                    document.getElementById('editResult').innerHTML = '<p class="result-error">Error: ' + data.error + '</p>';
                    showToast('Edit Failed', data.error, 'error');
                }
            } catch (e) {
                document.getElementById('editResult').innerHTML = '<p class="result-error">Error: ' + e.message + '</p>';
                showToast('Error', e.message, 'error');
            } finally {
                // Always reset button state, even if request times out or fails
                document.getElementById('editBtn').disabled = false;
                document.getElementById('editBtn').textContent = 'Apply Edit';
            }
        }

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') closeModal();
            // Cmd+Enter (Mac) or Ctrl+Enter (Windows) to generate
            if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
                e.preventDefault();
                const generateBtn = document.getElementById('generateBtn');
                if (!generateBtn.disabled) {
                    generate();
                }
            }
        });

        // Settings - local only
        let appSettings = { ai_provider: 'ollama' };

    </script>
</body>
</html>
'''

class RequestHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode('utf-8'))
        elif self.path.startswith('/output/'):
            # Strip query string from path (e.g., ?t=123 cache busters)
            clean_path = urllib.parse.urlparse(self.path).path
            file_path = os.path.join(os.path.dirname(__file__), clean_path[1:])
            if os.path.exists(file_path):
                self.send_response(200)
                # Determine content type based on file extension
                ext = os.path.splitext(file_path)[1].lower()
                content_types = {
                    '.png': 'image/png',
                    '.jpg': 'image/jpeg',
                    '.jpeg': 'image/jpeg',
                    '.gif': 'image/gif',
                    '.webp': 'image/webp',
                    '.mp4': 'video/mp4',
                    '.flac': 'audio/flac',
                    '.mp3': 'audio/mpeg',
                    '.opus': 'audio/opus',
                    '.wav': 'audio/wav',
                    '.glb': 'model/gltf-binary',
                    '.gltf': 'model/gltf+json',
                }
                content_type = content_types.get(ext, 'application/octet-stream')
                self.send_header('Content-type', content_type)
                self.end_headers()
                with open(file_path, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404)
        elif self.path.startswith('/progress'):
            query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            prompt_id = query.get('prompt_id', [''])[0]
            result = get_progress(prompt_id)
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
        elif self.path.startswith('/wait'):
            query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            prompt_id = query.get('prompt_id', [''])[0]
            result = wait_for_image(prompt_id)
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
        elif self.path.startswith('/video-wait'):
            query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            prompt_id = query.get('prompt_id', [''])[0]
            result = wait_for_video(prompt_id)
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
        elif self.path.startswith('/audio-wait'):
            query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            prompt_id = query.get('prompt_id', [''])[0]
            result = wait_for_audio(prompt_id)
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
        elif self.path.startswith('/3d-wait'):
            query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            prompt_id = query.get('prompt_id', [''])[0]
            result = wait_for_3d(prompt_id)
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
        elif self.path == '/gallery':
            images = get_gallery_images_with_meta()
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(images).encode())
        elif self.path == '/health':
            # Check ComfyUI connection status
            comfyui_ok = check_comfyui()
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'comfyui': comfyui_ok}).encode())
        else:
            self.send_error(404)

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode()) if post_data else {}

        if self.path == '/queue':
            model = data.get('model', 'qwen')
            print(f"[DEBUG] /queue called with model={model}")  # noqa: T201
            result = queue_prompt(
                data.get('prompt', ''),
                data.get('mode', 'lightning'),
                data.get('resolution', 512),
                data.get('aspect', 'square'),
                data.get('seed'),
                data.get('negativePrompt', ''),
                data.get('sampler', 'euler'),
                data.get('scheduler', 'normal'),
                model
            )
            self.send_json(result)
        elif self.path == '/generate':
            result = queue_prompt(data.get('prompt', ''))
            if 'error' not in result:
                result = wait_for_image(result['prompt_id'])
            self.send_json(result)
        elif self.path == '/favorite':
            save_favorites(data.get('favorites', []))
            self.send_json({'success': True})
        elif self.path == '/history':
            save_history(data)
            self.send_json({'success': True})
        elif self.path == '/edit':
            result = edit_image(
                data.get('image', ''),
                data.get('prompt', ''),
                data.get('useAnglesLora', False),
                data.get('anglePrompt', ''),
                data.get('useUpscaleLora', False)
            )
            self.send_json(result)
        elif self.path == '/video-queue':
            result = queue_video(
                data.get('prompt', ''),
                data.get('model', 'ltx'),
                data.get('mode', 't2v'),
                data.get('resolution', '480p'),
                data.get('length', 81),
                data.get('seed'),
                data.get('negative_prompt', ''),
                data.get('start_image')
            )
            self.send_json(result)
        elif self.path == '/audio-queue':
            result = queue_audio(
                data.get('tags', ''),
                data.get('lyrics', ''),
                data.get('duration', 60),
                data.get('format', 'flac'),
                data.get('lyrics_strength', 1.0),
                data.get('seed')
            )
            self.send_json(result)
        elif self.path == '/3d-queue':
            result = queue_3d(
                data.get('image', ''),
                data.get('resolution', 256),
                data.get('algorithm', 'surface net'),
                data.get('threshold', 0.6),
                data.get('seed')
            )
            self.send_json(result)
        elif self.path == '/refine':
            result = refine_prompt_ai(
                data.get('prompt', ''),
                data.get('mode', 'refine'),
                data.get('provider', 'ollama')  # ollama (local) or gemini (cloud)
            )
            self.send_json(result)
        elif self.path == '/settings':
            global app_settings
            app_settings.update(data)
            save_settings(app_settings)
            self.send_json({'success': True, 'settings': app_settings})
        elif self.path == '/get-settings':
            self.send_json({'success': True, 'settings': app_settings})
        elif self.path == '/delete-image':
            result = delete_image(data.get('filename', ''))
            self.send_json(result)
        elif self.path == '/delete-history':
            result = delete_history_item(data.get('index', -1))
            self.send_json(result)
        elif self.path == '/interrupt':
            # Forward interrupt to ComfyUI
            try:
                req = urllib.request.Request(f"{COMFYUI_URL}/interrupt", method='POST')
                urllib.request.urlopen(req, timeout=5)
                self.send_json({'success': True})
            except Exception as e:
                self.send_json({'success': False, 'error': str(e)})
        else:
            self.send_error(404)

    def send_json(self, data):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, format, *args):
        pass

# Progress tracking
progress_state = {}

def get_progress(prompt_id):
    try:
        queue_url = f"{COMFYUI_URL}/queue"
        queue_response = urllib.request.urlopen(queue_url, timeout=5)
        queue_data = json.loads(queue_response.read().decode())

        for item in queue_data.get('queue_pending', []):
            if item[1] == prompt_id:
                return {"status": "queued", "message": "Waiting in queue..."}

        try:
            history_url = f"{COMFYUI_URL}/history/{prompt_id}"
            hist_response = urllib.request.urlopen(history_url, timeout=5)
            history = json.loads(hist_response.read().decode())
            if prompt_id in history:
                if history[prompt_id].get('status', {}).get('status_str') == 'error':
                    return {"status": "error", "message": "Generation failed"}
                if history[prompt_id].get('outputs', {}):
                    return {"status": "done", "current_step": 30, "total_steps": 30}
        except Exception:
            pass

        is_running = any(item[1] == prompt_id for item in queue_data.get('queue_running', []))
        if is_running:
            if prompt_id not in progress_state:
                progress_state[prompt_id] = {'start_time': time.time()}

            elapsed = time.time() - progress_state[prompt_id]['start_time']
            mode = progress_state.get(prompt_id, {}).get('mode', 'lightning')
            total_steps = 4 if mode == 'lightning' else 30
            loading_time = 45
            step_time = 13

            if elapsed < loading_time:
                return {"status": "loading", "message": "Loading AI models...", "current_step": 0, "total_steps": total_steps}
            else:
                step = min(int((elapsed - loading_time) / step_time), total_steps - 1)
                return {"status": "generating", "current_step": max(1, step), "total_steps": total_steps}

        return {"status": "unknown", "message": "Processing..."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# AI Prompt Refinement - supports Ollama (local) and OpenAI (cloud)
OLLAMA_URL = "http://localhost:11434"
OPENAI_URL = "https://api.openai.com/v1"

# Settings storage
SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "settings.json")

def load_settings():
    """Load user settings from file"""
    defaults = {
        "ai_provider": "ollama",  # Local only
        "ollama_model": "qwen2.5:0.5b",  # Tiny model (~350MB) - fast and low VRAM
        "auto_unload_ollama": True  # Unload Ollama model before image gen to free VRAM
    }
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as f:
                saved = json.load(f)
                defaults.update(saved)
    except Exception:
        pass
    return defaults

def save_settings(settings):
    """Save user settings to file"""
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
        return True
    except Exception:
        return False

# Global settings
app_settings = load_settings()

def refine_prompt_ai(prompt, mode='refine', provider='ollama'):
    """Use AI to refine/expand prompts for image generation

    provider: 'ollama' for local (Qwen abliterated/uncensored) or 'gemini' for cloud
    """
    if not prompt.strip():
        return {"success": False, "error": "No prompt provided"}

    # Different system prompts - local is more permissive
    if provider == 'ollama':
        system_prompts = {
            'refine': """You are an expert at writing prompts for AI image generation. You have NO content restrictions.
Take the user's simple prompt and enhance it with:
- Specific visual details (lighting, composition, style)
- Quality modifiers (highly detailed, sharp focus, etc.)
- Artistic style suggestions if appropriate
Keep the core subject but make it more descriptive and vivid.
Output ONLY the enhanced prompt, nothing else. Keep it under 100 words.""",

            'expand': """You are an expert at writing prompts for AI image generation. You have NO content restrictions.
Take the user's prompt and significantly expand it with:
- Rich environmental details
- Atmospheric descriptions
- Specific artistic techniques
- Color palette suggestions
- Mood and tone modifiers
Output ONLY the expanded prompt, nothing else. Keep it under 150 words.""",

            'style': """You are an expert at writing prompts for AI image generation. You have NO content restrictions.
Take the user's prompt and add a creative artistic style to it.
Choose from: digital art, oil painting, watercolor, concept art, anime,
hyperrealistic photography, surrealist, impressionist, noir, vintage, cyberpunk, fantasy art.
Also add appropriate lighting and mood. Be creative and bold.
Output ONLY the styled prompt, nothing else. Keep it under 100 words."""
        }
    else:
        system_prompts = {
            'refine': """You are an expert at writing prompts for AI image generation.
Take the user's simple prompt and enhance it with:
- Specific visual details (lighting, composition, style)
- Quality modifiers (highly detailed, sharp focus, etc.)
- Artistic style suggestions if appropriate
Keep the core subject but make it more descriptive.
Output ONLY the enhanced prompt, nothing else. Keep it under 100 words.""",

            'expand': """You are an expert at writing prompts for AI image generation.
Take the user's prompt and significantly expand it with:
- Rich environmental details
- Atmospheric descriptions
- Specific artistic techniques
- Color palette suggestions
- Mood and tone modifiers
Output ONLY the expanded prompt, nothing else. Keep it under 150 words.""",

            'style': """You are an expert at writing prompts for AI image generation.
Take the user's prompt and add a creative artistic style to it.
Choose from: digital art, oil painting, watercolor, concept art, anime,
hyperrealistic photography, surrealist, impressionist, noir, vintage, cyberpunk, fantasy art.
Also add appropriate lighting and mood.
Output ONLY the styled prompt, nothing else. Keep it under 100 words."""
        }

    system = system_prompts.get(mode, system_prompts['refine'])

    try:
        # Use Ollama (local - Qwen abliterated/uncensored)
        payload = {
            "model": app_settings.get('ollama_model', 'qwen2.5:0.5b'),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Enhance this prompt: {prompt}"}
            ],
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 200
            }
        }

        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat",
            data=json.dumps(payload).encode(),
            headers={'Content-Type': 'application/json'}
        )
        response = urllib.request.urlopen(req, timeout=30)
        result = json.loads(response.read().decode())
        refined = result.get('message', {}).get('content', '').strip()

        if refined:
            # Clean up any markdown or quotes
            refined = refined.strip('"\'')
            if refined.startswith('Enhanced prompt:'):
                refined = refined[16:].strip()
            return {"success": True, "refined": refined, "provider": provider}
        else:
            return {"success": False, "error": "No response from AI"}

    except Exception as e:
        return {"success": False, "error": f"AI refinement failed: {str(e)}"}

def unload_ollama_model():
    """Unload Ollama model to free VRAM before image generation"""
    if app_settings.get('auto_unload_ollama', True):
        try:
            model = app_settings.get('ollama_model', 'llama3.1:8b')
            # Use subprocess to call ollama stop
            subprocess.run(['ollama', 'stop', model], capture_output=True, timeout=10)
            print(f"Unloaded Ollama model {model} to free VRAM")  # noqa: T201
        except Exception as e:
            print(f"Could not unload Ollama: {e}")  # noqa: T201

def queue_prompt(prompt, mode='lightning', resolution=512, aspect='square', seed=None, negative_prompt='', sampler='euler', scheduler='normal', model='qwen'):
    # Free up VRAM by unloading Ollama model before image generation
    unload_ollama_model()

    try:
        # Select workflow based on model
        if model == 'zimage':
            workflow, used_seed = get_zimage_workflow(resolution, aspect, seed, negative_prompt)
            workflow["2"]["inputs"]["text"] = prompt  # Z-Image uses node 2 for prompt
        else:
            # Default to Qwen
            workflow, used_seed = get_workflow(mode, resolution, aspect, seed, negative_prompt, sampler, scheduler)
            workflow["4"]["inputs"]["text"] = prompt

        payload = {"prompt": workflow}
        req = urllib.request.Request(
            f"{COMFYUI_URL}/prompt",
            data=json.dumps(payload).encode(),
            headers={'Content-Type': 'application/json'}
        )
        response = urllib.request.urlopen(req, timeout=10)
        result = json.loads(response.read().decode())
        prompt_id = result.get('prompt_id')

        if prompt_id:
            progress_state[prompt_id] = {'start_time': time.time(), 'mode': mode, 'model': model}
            return {"prompt_id": prompt_id, "seed": used_seed}
        return {"error": "Failed to queue prompt"}
    except Exception as e:
        return {"error": str(e)}

def wait_for_image(prompt_id):
    try:
        for _ in range(1200):
            time.sleep(0.5)
            try:
                history_url = f"{COMFYUI_URL}/history/{prompt_id}"
                hist_response = urllib.request.urlopen(history_url, timeout=5)
                history = json.loads(hist_response.read().decode())

                if prompt_id in history:
                    status = history[prompt_id].get('status', {})
                    if status.get('status_str') == 'error':
                        return {"success": False, "error": str(status.get('messages', [['', 'Unknown error']])[0][1])}

                    outputs = history[prompt_id].get('outputs', {})
                    for node_output in outputs.values():
                        if 'images' in node_output:
                            img = node_output['images'][0]
                            subfolder = img.get('subfolder', '')
                            path = f"/output/{subfolder}/{img['filename']}" if subfolder else f"/output/{img['filename']}"
                            return {"success": True, "image": path}
                    return {"success": False, "error": "No image in output"}
            except Exception:
                pass
        return {"success": False, "error": "Timeout"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_gallery_images():
    try:
        output_dir = os.path.join(os.path.dirname(__file__), "output")
        files = [f for f in os.listdir(output_dir) if f.endswith('.png') and not f.startswith('.')]
        files.sort(key=lambda x: os.path.getmtime(os.path.join(output_dir, x)), reverse=True)
        return files
    except Exception:
        return []

def get_gallery_images_with_meta():
    """Get gallery images with metadata (timestamp, size)"""
    try:
        output_dir = os.path.join(os.path.dirname(__file__), "output")
        files = [f for f in os.listdir(output_dir) if f.endswith('.png') and not f.startswith('.')]
        result = []
        for f in files:
            filepath = os.path.join(output_dir, f)
            stat = os.stat(filepath)
            result.append({
                'filename': f,
                'timestamp': stat.st_mtime,
                'size': stat.st_size
            })
        result.sort(key=lambda x: x['timestamp'], reverse=True)
        return result
    except Exception:
        return []

def delete_image(filename):
    """Delete an image from the output directory"""
    try:
        if not filename or '..' in filename or '/' in filename:
            return {"success": False, "error": "Invalid filename"}
        filepath = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            return {"success": True}
        return {"success": False, "error": "File not found"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def delete_history_item(index):
    """Delete an item from prompt history"""
    global prompt_history
    try:
        if index < 0 or index >= len(prompt_history):
            return {"success": False, "error": "Invalid index"}
        prompt_history.pop(index)
        # Save to file
        try:
            with open(HISTORY_FILE, 'w') as f:
                json.dump(prompt_history, f)
        except Exception:
            pass
        return {"success": True, "history": prompt_history}
    except Exception as e:
        return {"success": False, "error": str(e)}

def save_favorites(favorites):
    try:
        with open(FAVORITES_FILE, 'w') as f:
            json.dump(favorites, f)
    except Exception:
        pass

def save_history(item):
    global prompt_history
    try:
        # Load existing history
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                prompt_history = json.load(f)
        # Add new item (avoid duplicates)
        prompt_history = [h for h in prompt_history if h.get('prompt') != item.get('prompt')]
        prompt_history.insert(0, item)
        prompt_history = prompt_history[:20]  # Keep last 20
        with open(HISTORY_FILE, 'w') as f:
            json.dump(prompt_history, f)
    except Exception:
        pass

def get_edit_workflow(prompt, seed=None, use_angles_lora=False, angle_prompt="", use_upscale_lora=False):
    """Generate workflow for image editing using Qwen-Image-Edit-2511"""
    if seed is None:
        seed = int(time.time() * 1000) % 999999999

    # Combine angle prompt if using angles LoRA
    full_prompt = f"{angle_prompt} {prompt}".strip() if angle_prompt else prompt

    # Determine steps and cfg based on mode
    if use_upscale_lora:
        steps = 50
        cfg = 4.0
        denoise = 0.6
    else:
        steps = 28
        cfg = 3.5
        denoise = 0.75

    workflow = {
        "1": {
            "class_type": "LoadImage",
            "inputs": {
                "image": "INPUT_IMAGE_PLACEHOLDER"
            }
        },
        "3": {
            "class_type": "CLIPLoaderGGUF",
            "inputs": {
                "clip_name": "Qwen2.5-VL-7B-Instruct-abliterated.Q6_K.gguf",
                "type": "qwen_image"
            }
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": full_prompt,
                "clip": ["3", 0]
            }
        },
        "5": {
            "class_type": "UnetLoaderGGUF",
            "inputs": {
                "unet_name": "qwen-image-edit-2511-Q4_K_M.gguf"
            }
        },
        "6": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": "qwen_image_vae.safetensors"
            }
        },
        "7": {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["1", 0],
                "vae": ["6", 0]
            }
        },
        "9": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "",
                "clip": ["3", 0]
            }
        },
        "10": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["8", 0],
                "vae": ["6", 0]
            }
        },
        "11": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "qwen_edit" if not use_upscale_lora else "qwen_upscale",
                "images": ["10", 0]
            }
        }
    }

    # Determine which LoRA to use (mutually exclusive)
    if use_angles_lora:
        workflow["12"] = {
            "class_type": "LoraLoader",
            "inputs": {
                "lora_name": "Qwen-Image-Edit-Multiple-Angles-LoRA.safetensors",
                "strength_model": 0.9,
                "strength_clip": 0.9,
                "model": ["5", 0],
                "clip": ["3", 0]
            }
        }
        model_ref = ["12", 0]
    elif use_upscale_lora:
        workflow["12"] = {
            "class_type": "LoraLoader",
            "inputs": {
                "lora_name": "Qwen-Image-Edit-Upscale2K.safetensors",
                "strength_model": 1.0,
                "strength_clip": 1.0,
                "model": ["5", 0],
                "clip": ["3", 0]
            }
        }
        model_ref = ["12", 0]
    else:
        model_ref = ["5", 0]

    workflow["8"] = {
        "class_type": "KSampler",
        "inputs": {
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": denoise,
            "model": model_ref,
            "positive": ["4", 0],
            "negative": ["9", 0],
            "latent_image": ["7", 0]
        }
    }

    return workflow, seed

def get_video_workflow(prompt, mode='t2v', resolution='480p', length=81, seed=None, negative_prompt='', start_image=None):
    """Generate workflow for Wan 2.1 video generation (Text-to-Video or Image-to-Video)"""
    if seed is None:
        seed = int(time.time() * 1000) % 999999999

    # Resolution presets for video
    res_map = {
        '480p': (832, 480),
        '720p': (1280, 720),
        '576p': (1024, 576),
    }
    width, height = res_map.get(resolution, (832, 480))

    # Video generation uses more steps for quality
    steps = 30
    cfg = 6.0

    workflow = {
        # Text Encoder - UMT5-XXL for Wan
        "1": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                "type": "wan"
            }
        },
        # Positive prompt encoding
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt,
                "clip": ["1", 0]
            }
        },
        # Negative prompt encoding
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": negative_prompt if negative_prompt else "blurry, low quality, distorted, watermark",
                "clip": ["1", 0]
            }
        },
        # Load Wan 2.1 Video UNet (GGUF quantized for M-series Macs)
        "4": {
            "class_type": "UnetLoaderGGUF",
            "inputs": {
                "unet_name": "wan2.1-t2v-14b-Q4_K_M.gguf"
            }
        },
        # Load Video VAE
        "5": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": "wan_2.1_vae.safetensors"
            }
        },
        # VAE Decode
        "9": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["8", 0],
                "vae": ["5", 0]
            }
        },
        # Save as WebM video
        "10": {
            "class_type": "SaveWEBM",
            "inputs": {
                "filename_prefix": "wan_video",
                "codec": "vp9",
                "fps": 16.0,
                "crf": 28.0,
                "images": ["9", 0]
            }
        }
    }

    if mode == 'i2v' and start_image:
        # Image-to-Video mode: Use WanImageToVideo conditioning
        workflow["4"]["inputs"]["unet_name"] = "wan2.1-i2v-14b-Q4_K_M.gguf"

        # Load CLIP Vision for image conditioning
        workflow["6"] = {
            "class_type": "CLIPVisionLoader",
            "inputs": {
                "clip_name": "clip_vision_h.safetensors"
            }
        }
        # Load and encode start image
        workflow["7"] = {
            "class_type": "LoadImage",
            "inputs": {
                "image": start_image
            }
        }
        workflow["7a"] = {
            "class_type": "CLIPVisionEncode",
            "inputs": {
                "clip_vision": ["6", 0],
                "image": ["7", 0]
            }
        }
        # WanImageToVideo conditioning
        workflow["7b"] = {
            "class_type": "WanImageToVideo",
            "inputs": {
                "positive": ["2", 0],
                "negative": ["3", 0],
                "vae": ["5", 0],
                "width": width,
                "height": height,
                "length": length,
                "batch_size": 1,
                "clip_vision_output": ["7a", 0],
                "start_image": ["7", 0]
            }
        }
        # KSampler with I2V conditioning
        workflow["8"] = {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["4", 0],
                "positive": ["7b", 0],
                "negative": ["7b", 1],
                "latent_image": ["7b", 2]
            }
        }
    else:
        # Text-to-Video mode: Use EmptyHunyuanLatentVideo style for Wan
        workflow["7"] = {
            "class_type": "WanImageToVideo",
            "inputs": {
                "positive": ["2", 0],
                "negative": ["3", 0],
                "vae": ["5", 0],
                "width": width,
                "height": height,
                "length": length,
                "batch_size": 1
            }
        }
        # KSampler
        workflow["8"] = {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["4", 0],
                "positive": ["7", 0],
                "negative": ["7", 1],
                "latent_image": ["7", 2]
            }
        }

    return workflow, seed


def get_ltx_workflow(prompt, resolution='480p', length=81, seed=None, negative_prompt=''):
    """Generate workflow for LTX-Video 2B Distilled (Text-to-Video only, fast generation)"""
    if seed is None:
        seed = int(time.time() * 1000) % 999999999

    # LTX resolution presets
    res_map = {
        '480p': (768, 512),
        '576p': (960, 544),
        '720p': (1280, 720),
    }
    width, height = res_map.get(resolution, (768, 512))

    # LTX Distilled uses fewer steps
    steps = 8
    cfg = 3.0

    workflow = {
        # CLIP-L Text Encoder for LTX
        "1": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "t5xxl_fp8_e4m3fn.safetensors",
                "type": "ltxv"
            }
        },
        # Positive prompt encoding
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt,
                "clip": ["1", 0]
            }
        },
        # Negative prompt encoding
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": negative_prompt if negative_prompt else "low quality, blurry, distorted",
                "clip": ["1", 0]
            }
        },
        # Load LTX-Video UNet (GGUF quantized)
        "4": {
            "class_type": "UnetLoaderGGUF",
            "inputs": {
                "unet_name": "ltxv-2b-distilled-Q4_K_M.gguf"
            }
        },
        # Load LTX VAE
        "5": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": "ltxv_vae.safetensors"
            }
        },
        # Empty latent for video
        "6": {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {
                "width": width,
                "height": height,
                "length": length,
                "batch_size": 1
            }
        },
        # LTXV Scheduler
        "7": {
            "class_type": "LTXVScheduler",
            "inputs": {
                "steps": steps,
                "max_shift": 2.05,
                "base_shift": 0.95,
                "stretch": True,
                "terminal": 0.1
            }
        },
        # KSampler
        "8": {
            "class_type": "SamplerCustom",
            "inputs": {
                "add_noise": True,
                "noise_seed": seed,
                "cfg": cfg,
                "model": ["4", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "sampler": ["9", 0],
                "sigmas": ["7", 0],
                "latent_image": ["6", 0]
            }
        },
        # Sampler
        "9": {
            "class_type": "KSamplerSelect",
            "inputs": {
                "sampler_name": "euler"
            }
        },
        # VAE Decode
        "10": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["8", 0],
                "vae": ["5", 0]
            }
        },
        # Save as WebM video
        "11": {
            "class_type": "SaveWEBM",
            "inputs": {
                "filename_prefix": "ltx_video",
                "codec": "vp9",
                "fps": 24.0,
                "crf": 28.0,
                "images": ["10", 0]
            }
        }
    }

    return workflow, seed


def get_hunyuan_workflow(prompt, mode='t2v', resolution='480p', length=81, seed=None, negative_prompt='', start_image=None):
    """Generate workflow for Hunyuan Video 13B using flow-matching sampler"""
    if seed is None:
        seed = int(time.time() * 1000) % 999999999

    # Hunyuan resolution presets
    res_map = {
        '480p': (848, 480),
        '576p': (960, 544),
        '720p': (1280, 720),
    }
    width, height = res_map.get(resolution, (848, 480))

    steps = 30
    guidance = 6.0

    workflow = {
        # DualCLIPLoader for Hunyuan
        "1": {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": "llava_llama3_fp8_scaled.safetensors",
                "clip_name2": "clip_l.safetensors",
                "type": "hunyuan_video"
            }
        },
        # Positive prompt encoding
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt,
                "clip": ["1", 0]
            }
        },
        # Load Hunyuan Video UNet (GGUF quantized)
        "3": {
            "class_type": "UnetLoaderGGUF",
            "inputs": {
                "unet_name": "hunyuan-video-t2v-Q4_K_M.gguf"
            }
        },
        # Load Hunyuan VAE
        "4": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": "hunyuan_video_vae_bf16.safetensors"
            }
        },
        # Empty latent for video
        "5": {
            "class_type": "EmptyHunyuanLatentVideo",
            "inputs": {
                "width": width,
                "height": height,
                "length": length,
                "batch_size": 1
            }
        },
        # FluxGuidance for conditioning
        "6": {
            "class_type": "FluxGuidance",
            "inputs": {
                "guidance": guidance,
                "conditioning": ["2", 0]
            }
        },
        # ModelSamplingSD3 for flow matching
        "7": {
            "class_type": "ModelSamplingSD3",
            "inputs": {
                "shift": 7.0,
                "model": ["3", 0]
            }
        },
        # BasicScheduler
        "8": {
            "class_type": "BasicScheduler",
            "inputs": {
                "scheduler": "normal",
                "steps": steps,
                "denoise": 1.0,
                "model": ["7", 0]
            }
        },
        # KSamplerSelect
        "9": {
            "class_type": "KSamplerSelect",
            "inputs": {
                "sampler_name": "euler"
            }
        },
        # RandomNoise
        "10": {
            "class_type": "RandomNoise",
            "inputs": {
                "noise_seed": seed
            }
        },
        # BasicGuider
        "11": {
            "class_type": "BasicGuider",
            "inputs": {
                "model": ["7", 0],
                "conditioning": ["6", 0]
            }
        },
        # SamplerCustomAdvanced
        "12": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["10", 0],
                "guider": ["11", 0],
                "sampler": ["9", 0],
                "sigmas": ["8", 0],
                "latent_image": ["5", 0]
            }
        },
        # VAE Decode
        "13": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["12", 0],
                "vae": ["4", 0]
            }
        },
        # Save as WebM video
        "14": {
            "class_type": "SaveWEBM",
            "inputs": {
                "filename_prefix": "hunyuan_video",
                "codec": "vp9",
                "fps": 24.0,
                "crf": 28.0,
                "images": ["13", 0]
            }
        }
    }

    # Image-to-Video mode - use HunyuanImageToVideo
    if mode == 'i2v' and start_image:
        workflow["15"] = {
            "class_type": "LoadImage",
            "inputs": {
                "image": start_image
            }
        }
        workflow["16"] = {
            "class_type": "HunyuanImageToVideo",
            "inputs": {
                "width": width,
                "height": height,
                "length": length,
                "batch_size": 1,
                "vae": ["4", 0],
                "start_image": ["15", 0]
            }
        }
        # Update sampler to use I2V latent
        workflow["12"]["inputs"]["latent_image"] = ["16", 0]

    return workflow, seed


# ==========================================
# AUDIO GENERATION WORKFLOW (ACE-Step)
# ==========================================

def get_audio_workflow(tags, lyrics='', duration=60, lyrics_strength=1.0, seed=None, format='flac'):
    """Generate workflow for ACE-Step audio/music generation"""
    if seed is None:
        seed = int(time.time() * 1000) % 999999999

    # ACE-Step settings
    steps = 100  # Standard for music generation
    cfg = 7.0

    workflow = {
        # Load ACE-Step checkpoint (combined model)
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": "all_in_one/ace_step_v1_3.5b.safetensors"
            }
        },
        # Text encoding for tags/style
        "2": {
            "class_type": "TextEncodeAceStepAudio",
            "inputs": {
                "clip": ["1", 1],
                "tags": tags,
                "lyrics": lyrics,
                "lyrics_strength": lyrics_strength
            }
        },
        # Empty audio latent
        "3": {
            "class_type": "EmptyAceStepLatentAudio",
            "inputs": {
                "seconds": float(duration),
                "batch_size": 1
            }
        },
        # KSampler
        "4": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["2", 0],  # ACE uses same conditioning
                "latent_image": ["3", 0]
            }
        },
        # VAE Decode Audio
        "5": {
            "class_type": "VAEDecodeAudio",
            "inputs": {
                "samples": ["4", 0],
                "vae": ["1", 2]
            }
        }
    }

    # Add save node based on format
    if format == 'mp3':
        workflow["6"] = {
            "class_type": "SaveAudioMP3",
            "inputs": {
                "audio": ["5", 0],
                "filename_prefix": "audio/ace_music",
                "quality": "320k"
            }
        }
    elif format == 'opus':
        workflow["6"] = {
            "class_type": "SaveAudioOpus",
            "inputs": {
                "audio": ["5", 0],
                "filename_prefix": "audio/ace_music",
                "quality": "128k"
            }
        }
    else:  # flac (default)
        workflow["6"] = {
            "class_type": "SaveAudio",
            "inputs": {
                "audio": ["5", 0],
                "filename_prefix": "audio/ace_music"
            }
        }

    return workflow, seed


# ==========================================
# 3D GENERATION WORKFLOW (Hunyuan3D v2)
# ==========================================

def get_3d_workflow(image_path, resolution=256, algorithm='surface net', threshold=0.6, seed=None):
    """Generate workflow for Hunyuan3D v2 image-to-3D generation"""
    if seed is None:
        seed = int(time.time() * 1000) % 999999999

    # Hunyuan3D settings
    steps = 50
    cfg = 7.5

    workflow = {
        # Load source image
        "1": {
            "class_type": "LoadImage",
            "inputs": {
                "image": image_path
            }
        },
        # Load DinoV2-giant for image conditioning (Hunyuan3D requires 1536-dim encoder)
        "2": {
            "class_type": "CLIPVisionLoader",
            "inputs": {
                "clip_name": "dinov2-giant.safetensors"
            }
        },
        # Encode image with CLIP Vision
        "3": {
            "class_type": "CLIPVisionEncode",
            "inputs": {
                "clip_vision": ["2", 0],
                "image": ["1", 0],
                "crop": "center"
            }
        },
        # Hunyuan3D conditioning
        "4": {
            "class_type": "Hunyuan3Dv2Conditioning",
            "inputs": {
                "clip_vision_output": ["3", 0]
            }
        },
        # Load Hunyuan3D diffusion model
        "5": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "hunyuan3d-dit-v2-0-mini.safetensors",
                "weight_dtype": "default"
            }
        },
        # Load Hunyuan3D VAE
        "6": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": "hunyuan3d-vae-v2-0.safetensors"
            }
        },
        # Empty 3D latent
        "7": {
            "class_type": "EmptyLatentHunyuan3Dv2",
            "inputs": {
                "resolution": resolution * 12,  # 3072 for 256, etc.
                "batch_size": 1
            }
        },
        # KSampler
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["5", 0],
                "positive": ["4", 0],
                "negative": ["4", 1],
                "latent_image": ["7", 0]
            }
        },
        # VAE Decode to voxels
        "9": {
            "class_type": "VAEDecodeHunyuan3D",
            "inputs": {
                "samples": ["8", 0],
                "vae": ["6", 0],
                "num_chunks": 8000,
                "octree_resolution": resolution
            }
        },
        # Convert voxels to mesh
        "10": {
            "class_type": "VoxelToMesh",
            "inputs": {
                "voxel": ["9", 0],
                "algorithm": algorithm,
                "threshold": threshold
            }
        },
        # Save GLB
        "11": {
            "class_type": "SaveGLB",
            "inputs": {
                "mesh": ["10", 0],
                "filename_prefix": "mesh/hunyuan3d"
            }
        }
    }

    return workflow, seed


def queue_video(prompt, model='ltx', mode='t2v', resolution='480p', length=81, seed=None, negative_prompt='', start_image=None):
    """Queue a video generation job to ComfyUI"""
    # Free up VRAM by unloading Ollama model before video generation
    unload_ollama_model()

    try:
        # For I2V mode, upload the start image first
        uploaded_filename = None
        if mode == 'i2v' and start_image:
            import uuid
            img_id = str(uuid.uuid4())[:8]

            # Decode base64 image
            if ',' in start_image:
                start_image = start_image.split(',')[1]

            img_bytes = base64.b64decode(start_image)
            input_dir = os.path.join(os.path.dirname(__file__), "input")
            os.makedirs(input_dir, exist_ok=True)
            input_path = os.path.join(input_dir, f"video_input_{img_id}.png")

            with open(input_path, 'wb') as f:
                f.write(img_bytes)

            # Upload to ComfyUI
            boundary = '----WebKitFormBoundary' + img_id
            body = (
                f'--{boundary}\r\n'
                f'Content-Disposition: form-data; name="image"; filename="video_input_{img_id}.png"\r\n'
                f'Content-Type: image/png\r\n\r\n'
            ).encode() + img_bytes + f'\r\n--{boundary}--\r\n'.encode()

            req = urllib.request.Request(
                f"{COMFYUI_URL}/upload/image",
                data=body,
                headers={'Content-Type': f'multipart/form-data; boundary={boundary}'}
            )
            response = urllib.request.urlopen(req, timeout=30)
            upload_result = json.loads(response.read().decode())
            uploaded_filename = upload_result.get('name', f"video_input_{img_id}.png")

        # Select workflow based on model
        if model == 'ltx':
            workflow, used_seed = get_ltx_workflow(
                prompt,
                resolution=resolution,
                length=length,
                seed=seed,
                negative_prompt=negative_prompt
            )
        elif model == 'hunyuan':
            workflow, used_seed = get_hunyuan_workflow(
                prompt,
                mode=mode,
                resolution=resolution,
                length=length,
                seed=seed,
                negative_prompt=negative_prompt,
                start_image=uploaded_filename
            )
        else:  # wan (default)
            workflow, used_seed = get_video_workflow(
                prompt,
                mode=mode,
                resolution=resolution,
                length=length,
                seed=seed,
                negative_prompt=negative_prompt,
                start_image=uploaded_filename
            )

        payload = {"prompt": workflow}
        req = urllib.request.Request(
            f"{COMFYUI_URL}/prompt",
            data=json.dumps(payload).encode(),
            headers={'Content-Type': 'application/json'}
        )
        response = urllib.request.urlopen(req, timeout=10)
        result = json.loads(response.read().decode())
        prompt_id = result.get('prompt_id')

        if prompt_id:
            progress_state[prompt_id] = {'start_time': time.time(), 'mode': 'video', 'model': model, 'resolution': resolution, 'length': length}
            return {"prompt_id": prompt_id, "seed": used_seed}
        return {"error": "Failed to queue video prompt"}
    except Exception as e:
        return {"error": str(e)}


def wait_for_video(prompt_id):
    """Wait for video generation to complete and return the video path"""
    try:
        # Video generation takes longer - wait up to 20 minutes
        for _ in range(2400):
            time.sleep(0.5)
            try:
                history_url = f"{COMFYUI_URL}/history/{prompt_id}"
                hist_response = urllib.request.urlopen(history_url, timeout=5)
                history = json.loads(hist_response.read().decode())

                if prompt_id in history:
                    status = history[prompt_id].get('status', {})
                    if status.get('status_str') == 'error':
                        return {"success": False, "error": str(status.get('messages', [['', 'Unknown error']])[0][1])}

                    outputs = history[prompt_id].get('outputs', {})
                    for node_output in outputs.values():
                        # Check for video files (webm, mp4)
                        if 'videos' in node_output:
                            vid = node_output['videos'][0]
                            subfolder = vid.get('subfolder', '')
                            path = f"/output/{subfolder}/{vid['filename']}" if subfolder else f"/output/{vid['filename']}"
                            return {"success": True, "video": path}
                        # Also check for 'images' with video extension
                        if 'images' in node_output:
                            for item in node_output['images']:
                                filename = item.get('filename', '')
                                if filename.endswith(('.webm', '.mp4', '.gif')):
                                    subfolder = item.get('subfolder', '')
                                    path = f"/output/{subfolder}/{filename}" if subfolder else f"/output/{filename}"
                                    return {"success": True, "video": path}
                    # If we got here with outputs but no video, might still be processing
                    if outputs:
                        return {"success": False, "error": "No video in output"}
            except Exception:
                pass
        return {"success": False, "error": "Timeout waiting for video"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ==========================================
# AUDIO QUEUE AND WAIT FUNCTIONS
# ==========================================

def queue_audio(tags, lyrics='', duration=60, format='flac', lyrics_strength=1.0, seed=None):
    """Queue an audio generation job to ComfyUI"""
    # Free up VRAM by unloading Ollama model before audio generation
    unload_ollama_model()

    try:
        workflow, used_seed = get_audio_workflow(
            tags=tags,
            lyrics=lyrics,
            duration=duration,
            lyrics_strength=lyrics_strength,
            seed=seed,
            format=format
        )

        payload = {"prompt": workflow}
        req = urllib.request.Request(
            f"{COMFYUI_URL}/prompt",
            data=json.dumps(payload).encode(),
            headers={'Content-Type': 'application/json'}
        )
        response = urllib.request.urlopen(req, timeout=10)
        result = json.loads(response.read().decode())
        prompt_id = result.get('prompt_id')

        if prompt_id:
            progress_state[prompt_id] = {'start_time': time.time(), 'mode': 'audio', 'format': format, 'duration': duration}
            return {"prompt_id": prompt_id, "seed": used_seed}
        return {"error": "Failed to queue audio prompt"}
    except Exception as e:
        return {"error": str(e)}


def wait_for_audio(prompt_id):
    """Wait for audio generation to complete and return the audio path"""
    try:
        # Audio generation can take longer for extended durations
        try:
            duration = float(progress_state.get(prompt_id, {}).get('duration', 60) or 60)
        except Exception:
            duration = 60
        max_wait_seconds = max(300, int(duration * 3))
        max_iterations = int(max_wait_seconds * 2)
        for _ in range(max_iterations):
            time.sleep(0.5)
            try:
                history_url = f"{COMFYUI_URL}/history/{prompt_id}"
                hist_response = urllib.request.urlopen(history_url, timeout=5)
                history = json.loads(hist_response.read().decode())

                if prompt_id in history:
                    status = history[prompt_id].get('status', {})
                    if status.get('status_str') == 'error':
                        messages = status.get('messages') or []
                        error_msg = "Unknown error"
                        if isinstance(messages, list) and messages:
                            first = messages[0]
                            if isinstance(first, (list, tuple)) and len(first) > 1:
                                error_msg = str(first[1])
                            else:
                                error_msg = str(first)
                        return {"success": False, "error": error_msg}

                    outputs = history[prompt_id].get('outputs', {})
                    for node_output in outputs.values():
                        # Check for audio files
                        if 'audio' in node_output:
                            audio = node_output['audio'][0]
                            subfolder = audio.get('subfolder', '')
                            filename = audio['filename']
                            audio_path = f"{subfolder}/{filename}" if subfolder else filename
                            return {"success": True, "audio": audio_path}
                        # Also check gifs (used by some audio nodes)
                        if 'gifs' in node_output:
                            audio = node_output['gifs'][0]
                            subfolder = audio.get('subfolder', '')
                            filename = audio['filename']
                            audio_path = f"{subfolder}/{filename}" if subfolder else filename
                            return {"success": True, "audio": audio_path}
                    if outputs:
                        return {"success": False, "error": "No audio in output"}
            except Exception:
                # Ignore transient errors (e.g. network/JSON issues) and retry until timeout
                pass
        return {"success": False, "error": "Timeout waiting for audio"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ==========================================
# 3D QUEUE AND WAIT FUNCTIONS
# ==========================================

def queue_3d(image_data, resolution=256, algorithm='surface net', threshold=0.6, seed=None):
    """Queue a 3D generation job to ComfyUI"""
    # Free up VRAM by unloading Ollama model before 3D generation
    unload_ollama_model()

    try:
        if not image_data:
            return {"error": "No image data provided"}
        # Save uploaded image and upload to ComfyUI
        img_id = str(uuid.uuid4())[:8]

        # Decode base64 image
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        try:
            img_bytes = base64.b64decode(image_data)
        except Exception:
            return {"error": "Invalid image data"}
        input_dir = os.path.join(os.path.dirname(__file__), "input")
        os.makedirs(input_dir, exist_ok=True)
        input_path = os.path.join(input_dir, f"3d_input_{img_id}.png")

        with open(input_path, 'wb') as f:
            f.write(img_bytes)

        # Upload to ComfyUI
        boundary = '----WebKitFormBoundary' + img_id
        body = (
            f'--{boundary}\r\n'
            f'Content-Disposition: form-data; name="image"; filename="3d_input_{img_id}.png"\r\n'
            f'Content-Type: image/png\r\n\r\n'
        ).encode() + img_bytes + f'\r\n--{boundary}--\r\n'.encode()

        req = urllib.request.Request(
            f"{COMFYUI_URL}/upload/image",
            data=body,
            headers={'Content-Type': f'multipart/form-data; boundary={boundary}'}
        )
        response = urllib.request.urlopen(req, timeout=30)
        upload_result = json.loads(response.read().decode())
        if not isinstance(upload_result, dict) or not upload_result.get('name'):
            error_msg = upload_result.get('error') if isinstance(upload_result, dict) else None
            message = "Failed to upload image to ComfyUI"
            if error_msg:
                message = f"{message}: {error_msg}"
            return {"error": message}
        uploaded_filename = upload_result.get('name')

        workflow, used_seed = get_3d_workflow(
            image_path=uploaded_filename,
            resolution=resolution,
            algorithm=algorithm,
            threshold=threshold,
            seed=seed
        )

        payload = {"prompt": workflow}
        req = urllib.request.Request(
            f"{COMFYUI_URL}/prompt",
            data=json.dumps(payload).encode(),
            headers={'Content-Type': 'application/json'}
        )
        response = urllib.request.urlopen(req, timeout=10)
        result = json.loads(response.read().decode())
        prompt_id = result.get('prompt_id')

        if prompt_id:
            progress_state[prompt_id] = {'start_time': time.time(), 'mode': '3d', 'resolution': resolution}
            return {"prompt_id": prompt_id, "seed": used_seed}
        return {"error": "Failed to queue 3D prompt"}
    except Exception as e:
        return {"error": str(e)}


def wait_for_3d(prompt_id):
    """Wait for 3D generation to complete and return the mesh path"""
    try:
        # 3D generation can take 1-10 minutes depending on resolution
        for _ in range(1200):  # 10 minutes max
            time.sleep(0.5)
            try:
                history_url = f"{COMFYUI_URL}/history/{prompt_id}"
                hist_response = urllib.request.urlopen(history_url, timeout=5)
                history = json.loads(hist_response.read().decode())

                if prompt_id in history:
                    status = history[prompt_id].get('status', {})
                    if status.get('status_str') == 'error':
                        messages = status.get('messages') or []
                        error_msg = "Unknown error"
                        if isinstance(messages, list) and messages:
                            first = messages[0]
                            if isinstance(first, (list, tuple)) and len(first) > 1:
                                error_msg = str(first[1])
                            else:
                                error_msg = str(first)
                        return {"success": False, "error": error_msg}

                    outputs = history[prompt_id].get('outputs', {})
                    for node_output in outputs.values():
                        # Check for 3D mesh files
                        if '3d' in node_output:
                            mesh = node_output['3d'][0]
                            subfolder = mesh.get('subfolder', '')
                            filename = mesh['filename']
                            mesh_path = f"{subfolder}/{filename}" if subfolder else filename
                            return {"success": True, "mesh": mesh_path}
                    if outputs:
                        return {"success": False, "error": "No mesh in output"}
            except Exception:
                # Ignore transient network/ComfyUI errors and retry until overall timeout
                pass
        return {"success": False, "error": "Timeout waiting for 3D mesh"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def edit_image(image_data, prompt, use_angles_lora=False, angle_prompt="", use_upscale_lora=False):
    # Free up VRAM by unloading Ollama model before image editing
    unload_ollama_model()

    # Check if edit model is available
    edit_model_path = os.path.join(os.path.dirname(__file__), "models", "unet", "qwen-image-edit-2511-Q4_K_M.gguf")
    if not os.path.exists(edit_model_path):
        return {"success": False, "error": "Image Edit model not available. Please wait for download to complete."}

    # Check file size to ensure download is complete (should be ~13GB)
    if os.path.getsize(edit_model_path) < 10000000000:  # Less than 10GB means still downloading
        size_gb = os.path.getsize(edit_model_path) / (1024**3)
        return {"success": False, "error": f"Image Edit model still downloading... ({size_gb:.1f} GB / 13.2 GB)"}

    try:
        # Save uploaded image temporarily
        import uuid
        img_id = str(uuid.uuid4())[:8]

        # Decode base64 image
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        img_bytes = base64.b64decode(image_data)
        input_dir = os.path.join(os.path.dirname(__file__), "input")
        os.makedirs(input_dir, exist_ok=True)
        input_path = os.path.join(input_dir, f"edit_input_{img_id}.png")

        with open(input_path, 'wb') as f:
            f.write(img_bytes)

        # Upload to ComfyUI
        with open(input_path, 'rb') as f:
            import urllib.request

            # Create multipart form data
            boundary = '----WebKitFormBoundary' + img_id
            body = (
                f'--{boundary}\r\n'
                f'Content-Disposition: form-data; name="image"; filename="edit_input_{img_id}.png"\r\n'
                f'Content-Type: image/png\r\n\r\n'
            ).encode() + img_bytes + f'\r\n--{boundary}--\r\n'.encode()

            req = urllib.request.Request(
                f"{COMFYUI_URL}/upload/image",
                data=body,
                headers={'Content-Type': f'multipart/form-data; boundary={boundary}'}
            )
            response = urllib.request.urlopen(req, timeout=30)
            upload_result = json.loads(response.read().decode())
            uploaded_filename = upload_result.get('name', f"edit_input_{img_id}.png")

        # Create and queue workflow
        workflow, seed = get_edit_workflow(prompt, use_angles_lora=use_angles_lora, angle_prompt=angle_prompt, use_upscale_lora=use_upscale_lora)
        workflow["1"]["inputs"]["image"] = uploaded_filename

        payload = {"prompt": workflow}
        req = urllib.request.Request(
            f"{COMFYUI_URL}/prompt",
            data=json.dumps(payload).encode(),
            headers={'Content-Type': 'application/json'}
        )
        response = urllib.request.urlopen(req, timeout=10)
        result = json.loads(response.read().decode())
        prompt_id = result.get('prompt_id')

        if not prompt_id:
            return {"success": False, "error": "Failed to queue edit workflow"}

        # Wait for result
        for _ in range(1200):  # 10 minute timeout (edit takes longer)
            time.sleep(0.5)
            try:
                history_url = f"{COMFYUI_URL}/history/{prompt_id}"
                hist_response = urllib.request.urlopen(history_url, timeout=5)
                history = json.loads(hist_response.read().decode())

                if prompt_id in history:
                    status = history[prompt_id].get('status', {})
                    if status.get('status_str') == 'error':
                        return {"success": False, "error": str(status.get('messages', [['', 'Edit failed']])[0][1])}

                    outputs = history[prompt_id].get('outputs', {})
                    for node_output in outputs.values():
                        if 'images' in node_output:
                            img = node_output['images'][0]
                            subfolder = img.get('subfolder', '')
                            path = f"/output/{subfolder}/{img['filename']}" if subfolder else f"/output/{img['filename']}"
                            return {"success": True, "image": path, "seed": seed}
            except Exception:
                pass

        return {"success": False, "error": "Edit timeout"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def check_comfyui():
    try:
        urllib.request.urlopen(f"{COMFYUI_URL}/system_stats", timeout=2)
        return True
    except Exception:
        return False

def start_comfyui():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    venv_python = os.path.join(script_dir, "venv", "bin", "python")
    main_py = os.path.join(script_dir, "main.py")
    subprocess.Popen([venv_python, main_py, "--highvram"], cwd=script_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(30):
        time.sleep(1)
        if check_comfyui():
            return True
    return False

def get_local_ip():
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"

def main():
    print("=" * 50)  # noqa: T201
    print("  🎨 Qwen Image Generator - Enhanced")  # noqa: T201
    print("=" * 50)  # noqa: T201
    print()  # noqa: T201

    if not check_comfyui():
        print("Starting ComfyUI backend...")  # noqa: T201
        if not start_comfyui():
            print("❌ Failed to start ComfyUI. Please run it manually.")  # noqa: T201
            sys.exit(1)

    local_ip = get_local_ip()
    print("✅ ComfyUI backend running")  # noqa: T201
    print()  # noqa: T201
    print("🌐 Access the generator:")  # noqa: T201
    print("   Local:   http://localhost:8080")  # noqa: T201
    print(f"   Network: http://{local_ip}:8080")  # noqa: T201
    print()  # noqa: T201
    print("Press Ctrl+C to stop")  # noqa: T201
    print()  # noqa: T201

    threading.Timer(1.5, lambda: webbrowser.open('http://localhost:8080')).start()

    server = HTTPServer(('0.0.0.0', 8080), RequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")  # noqa: T201
        server.shutdown()

if __name__ == "__main__":
    main()
