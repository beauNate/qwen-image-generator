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
from http.server import HTTPServer, SimpleHTTPRequestHandler
import subprocess

COMFYUI_URL = "http://127.0.0.1:8188"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
FAVORITES_FILE = os.path.join(os.path.dirname(__file__), "favorites.json")
HISTORY_FILE = os.path.join(os.path.dirname(__file__), "prompt_history.json")

# Store last used seed for regeneration
last_seeds = {}

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
            /* Spacing - Consistent 8px grid */
            --space-1: 4px;
            --space-2: 8px;
            --space-3: 12px;
            --space-4: 16px;
            --space-5: 20px;
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

            /* Semantic */
            --success: #30D158;
            --success-bg: rgba(48, 209, 88, 0.12);
            --warning: #FFD60A;
            --error: #FF453A;
            --error-bg: rgba(255, 69, 58, 0.12);

            /* Typography - Consistent scale */
            --font-sans: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", system-ui, sans-serif;
            --font-mono: "SF Mono", "Fira Code", ui-monospace, monospace;
            --text-xs: 11px;
            --text-sm: 13px;
            --text-base: 14px;
            --text-lg: 16px;
            --text-xl: 20px;
            --text-2xl: 28px;

            /* Radius - Apple's signature */
            --radius-sm: 8px;
            --radius-md: 12px;
            --radius-lg: 16px;
            --radius-xl: 20px;
            --radius-full: 9999px;

            /* Animation - Unified timing */
            --ease-out: cubic-bezier(0.25, 1, 0.5, 1);
            --ease-spring: cubic-bezier(0.34, 1.56, 0.64, 1);
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
            font-size: var(--text-base);
            line-height: 1.5;
            color: var(--text-primary);
            max-width: 600px;
            margin: 0 auto;
            padding: var(--space-4);
            background: linear-gradient(145deg, #1a1a2e 0%, #0d0d1a 50%, #1a0a2e 100%);
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

        /* Ambient background glow */
        body::before {
            content: '';
            position: fixed;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(ellipse at 30% 20%, rgba(99, 102, 241, 0.12) 0%, transparent 50%),
                        radial-gradient(ellipse at 70% 80%, rgba(168, 85, 247, 0.08) 0%, transparent 50%),
                        radial-gradient(ellipse at 50% 50%, rgba(10, 132, 255, 0.04) 0%, transparent 70%);
            pointer-events: none;
            z-index: -1;
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
            font-size: var(--text-2xl);
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
            font-size: var(--text-base);
            font-weight: 400;
            animation: fadeInUp 0.4s var(--ease-out) 0.15s both;
        }

        /* Connection Status */
        .connection-status {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: var(--text-xs);
            color: var(--text-quaternary);
            margin-left: 8px;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--text-quaternary);
            transition: background 0.3s;
        }
        .status-dot.connected { background: var(--success); box-shadow: 0 0 6px var(--success); }
        .status-dot.disconnected { background: var(--error); box-shadow: 0 0 6px var(--error); }
        .status-dot.checking { background: var(--warning); animation: pulse 1s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }

        /* Toast Notifications */
        .toast-container {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 10000;
            display: flex;
            flex-direction: column;
            gap: 10px;
            pointer-events: none;
        }
        .toast {
            background: var(--glass-bg);
            backdrop-filter: blur(20px) saturate(180%);
            -webkit-backdrop-filter: blur(20px) saturate(180%);
            border: 1px solid var(--glass-border);
            border-radius: var(--radius-md);
            padding: 12px 16px;
            display: flex;
            align-items: center;
            gap: 10px;
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
            font-size: 18px;
            flex-shrink: 0;
        }
        .toast-content {
            flex: 1;
        }
        .toast-title {
            font-weight: 600;
            font-size: var(--text-sm);
            color: var(--text-primary);
        }
        .toast-message {
            font-size: var(--text-xs);
            color: var(--text-secondary);
            margin-top: 2px;
        }
        .toast.success { border-left: 3px solid var(--success); }
        .toast.error { border-left: 3px solid var(--error); }
        .toast.warning { border-left: 3px solid var(--warning); }
        .toast.info { border-left: 3px solid var(--accent); }
        @keyframes toastIn {
            from { opacity: 0; transform: translateX(100px); }
            to { opacity: 1; transform: translateX(0); }
        }
        @keyframes toastOut {
            from { opacity: 1; transform: translateX(0); }
            to { opacity: 0; transform: translateX(100px); }
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
            font-size: var(--text-base);
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

        label {
            display: block;
            margin-bottom: var(--space-2);
            font-weight: 500;
            font-size: var(--text-sm);
            color: var(--text-secondary);
            letter-spacing: 0.01em;
        }

        textarea, input[type="text"], input[type="number"] {
            width: 100%;
            padding: var(--space-3) var(--space-4);
            border: 1px solid var(--glass-border);
            border-radius: var(--radius-md);
            font-size: var(--text-base);
            font-family: var(--font-sans);
            background: rgba(0, 0, 0, 0.2);
            color: var(--text-primary);
            margin-bottom: var(--space-3);
            transition: all var(--duration-base) var(--ease-out);
        }

        textarea { height: 110px; resize: vertical; overflow-y: auto; }

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
        .option-group { flex: 1; min-width: 130px; }
        .option-group label { display: block; margin-bottom: 4px; }
        .option-group .option-hint { display: block; margin-bottom: 6px; }

        select {
            width: 100%;
            padding: var(--space-2) var(--space-3);
            border: 1px solid var(--glass-border);
            border-radius: var(--radius-md);
            font-size: var(--text-base);
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
            font-size: var(--text-sm);
            color: var(--accent);
            margin-bottom: var(--space-3);
            font-weight: 500;
        }

        /* Buttons - Apple style with glow */
        button {
            padding: var(--space-2) var(--space-5);
            font-size: var(--text-base);
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
            transition: transform 0.1s var(--ease-out);
        }

        button:disabled {
            background: rgba(255, 255, 255, 0.08);
            color: var(--text-quaternary);
            cursor: not-allowed;
            box-shadow: none;
            transform: none;
        }

        .btn-row { display: flex; gap: var(--space-2); }
        .btn-row button { flex: 1; }

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

        .btn-green {
            background: var(--success);
            box-shadow: 0 2px 8px rgba(48, 209, 88, 0.35);
        }

        .btn-green:hover:not(:disabled) {
            background: #3adb62;
            box-shadow: 0 4px 16px rgba(48, 209, 88, 0.35);
        }

        /* Cancel button - subdued until hovered */
        .btn-cancel {
            background: rgba(255, 69, 58, 0.15);
            color: rgba(255, 69, 58, 0.8);
            border: 1px solid rgba(255, 69, 58, 0.3);
            box-shadow: none;
        }
        .btn-cancel:hover:not(:disabled) {
            background: rgba(255, 69, 58, 0.25);
            color: #FF453A;
            border-color: rgba(255, 69, 58, 0.5);
            box-shadow: none;
            transform: none;
        }

        /* Cheatsheet buttons */
        .cheat-btn {
            padding: 8px 16px;
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 6px;
            color: #fff;
            cursor: pointer;
            font-size: 13px;
            transition: all 0.15s ease-out;
            min-height: 44px;
        }
        .cheat-btn:hover {
            background: rgba(102, 126, 234, 0.25);
            border-color: rgba(102, 126, 234, 0.5);
        }

        /* Floating AI buttons - 44px touch targets */
        .ai-float-btn {
            width: 36px;
            height: 36px;
            min-width: 44px;
            min-height: 44px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 6px;
            color: #fff;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.15s ease-out;
            opacity: 0.7;
        }
        .ai-float-btn:hover {
            opacity: 1;
            background: rgba(102, 126, 234, 0.3);
            border-color: rgba(102, 126, 234, 0.5);
        }

        /* Status */
        #status {
            text-align: center;
            padding: var(--space-3) var(--space-4);
            margin: var(--space-3) 0;
            border-radius: var(--radius-md);
            display: none;
            font-size: var(--text-base);
            font-weight: 500;
            backdrop-filter: blur(10px);
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }

        .generating {
            background: var(--accent-bg);
            border: 1px solid rgba(10, 132, 255, 0.25);
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
            border: 1px solid rgba(255, 69, 58, 0.25);
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

        @keyframes progressGlow {
            0%, 100% { box-shadow: 0 0 8px var(--accent-glow); }
            50% { box-shadow: 0 0 16px var(--accent-glow), 0 0 24px var(--accent-glow); }
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--accent), #A855F7);
            border-radius: var(--radius-full);
            transition: width var(--duration-slow) var(--ease-out);
            animation: progressGlow 2s ease-in-out infinite;
        }

        .progress-text {
            display: flex;
            justify-content: space-between;
            margin-top: var(--space-1);
            font-size: var(--text-xs);
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
            font-size: var(--text-sm);
            color: var(--text-tertiary);
        }

        .seed-display {
            font-family: var(--font-mono);
            background: var(--glass-bg);
            padding: var(--space-1) var(--space-2);
            border-radius: var(--radius-sm);
            cursor: pointer;
            font-size: var(--text-xs);
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
            font-size: var(--text-xs);
            font-family: var(--font-mono);
            color: var(--text-quaternary);
            text-align: center;
            padding-bottom: var(--space-2);
        }

        .btn-sm {
            padding: var(--space-1) var(--space-2);
            font-size: var(--text-xs);
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
            font-size: var(--text-xs);
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
            transition: width 0.3s ease;
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
            font-size: var(--text-sm);
            background: rgba(0, 0, 0, 0.6);
            backdrop-filter: blur(8px);
            padding: 6px;
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
            font-size: var(--text-xs);
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
            font-size: var(--text-xs);
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
            font-size: var(--text-sm);
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
            font-size: 24px;
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

        /* Examples */
        .examples {
            margin-top: var(--space-4);
            padding: var(--space-4);
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            border-radius: var(--radius-lg);
            animation: fadeInUp 0.4s var(--ease-out) 0.3s both;
        }

        .examples h3 {
            margin: 0 0 var(--space-2) 0;
            font-size: var(--text-xs);
            color: var(--text-tertiary);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 600;
        }

        .example-btn {
            display: inline-block;
            padding: var(--space-1) var(--space-3);
            margin: var(--space-1);
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid var(--glass-border);
            border-radius: var(--radius-full);
            cursor: pointer;
            font-size: var(--text-sm);
            color: var(--text-secondary);
            transition: all var(--duration-base) var(--ease-spring);
        }

        .example-btn:hover {
            background: rgba(255, 255, 255, 0.12);
            border-color: var(--glass-border-hover);
            color: var(--text-primary);
            transform: translateY(-2px);
        }

        .example-btn:active {
            transform: scale(0.97) translateY(0);
        }

        /* Advanced toggle */
        .advanced-toggle {
            color: var(--text-tertiary);
            cursor: pointer;
            font-size: var(--text-sm);
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
            font-size: var(--text-xs);
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

        /* Filter tabs */
        .filter-tabs {
            display: flex;
            gap: var(--space-2);
            margin-bottom: var(--space-3);
            flex-wrap: wrap;
        }

        .filter-tab {
            padding: var(--space-1) var(--space-3);
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            border-radius: var(--radius-full);
            cursor: pointer;
            font-size: var(--text-sm);
            color: var(--text-secondary);
            transition: all var(--duration-base) var(--ease-spring);
        }

        .filter-tab:hover {
            background: var(--glass-bg-hover);
            color: var(--text-primary);
            transform: translateY(-1px);
        }

        .filter-tab:active {
            transform: scale(0.97);
        }

        .filter-tab.active {
            background: var(--accent-bg);
            border-color: rgba(10, 132, 255, 0.25);
            color: var(--accent);
        }

        /* Presets - Pill buttons */
        .presets {
            display: flex;
            gap: var(--space-1);
            margin-bottom: var(--space-3);
            overflow-x: auto;
            padding-bottom: 4px;
            -webkit-overflow-scrolling: touch;
            scrollbar-width: thin;
        }
        .presets::-webkit-scrollbar { height: 4px; }
        .presets::-webkit-scrollbar-track { background: transparent; }
        .presets::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.2); border-radius: 2px; }

        .preset-btn {
            padding: 8px 16px;
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            border-radius: 8px;
            cursor: pointer;
            font-size: 13px;
            color: var(--text-secondary);
            transition: all var(--duration-fast) var(--ease-out);
            white-space: nowrap;
            flex-shrink: 0;
            min-height: 44px;
            display: inline-flex;
            align-items: center;
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
            background: rgba(10, 132, 255, 0.25) !important;
            border: 2px solid var(--accent) !important;
            color: #fff !important;
            font-weight: 600;
            box-shadow: 0 0 0 1px rgba(10, 132, 255, 0.3);
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
            font-size: var(--text-sm);
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
            font-size: var(--text-sm);
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
            font-size: var(--text-sm);
        }

        .history-dropdown.show {
            display: block;
            animation: dropdownOpen 0.2s var(--ease-spring);
        }

        .history-item {
            padding: var(--space-3) var(--space-4);
            cursor: pointer;
            border-bottom: 1px solid var(--glass-border);
            font-size: var(--text-base);
            color: var(--text-secondary);
            transition: all var(--duration-base);
        }

        .history-item:hover {
            background: var(--glass-bg-hover);
            color: var(--text-primary);
        }

        .history-item:last-child { border-bottom: none; }

        .history-meta {
            font-size: var(--text-xs);
            color: var(--text-quaternary);
            margin-top: var(--space-1);
            font-family: var(--font-mono);
        }

        /* Batch options */
        .batch-row {
            display: flex;
            align-items: center;
            gap: var(--space-2);
            margin-bottom: var(--space-3);
        }

        .batch-label { font-size: var(--text-sm); color: var(--text-tertiary); }

        .batch-input {
            width: 80px !important;
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
            font-size: var(--text-sm);
            font-weight: 600;
            color: var(--text-secondary);
        }
        .queue-count {
            font-size: var(--text-xs);
            font-family: var(--font-mono);
            color: var(--accent);
            background: var(--accent-bg);
            padding: 2px 8px;
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
            font-size: var(--text-sm);
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
            font-size: var(--text-xs);
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
            font-size: var(--text-xs);
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
            font-size: var(--text-lg);
            font-weight: 600;
            color: var(--text-primary);
        }

        .split-compare-hint {
            font-size: var(--text-sm);
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
            font-size: var(--text-sm);
            border-radius: var(--radius-sm);
        }

        .split-panel-title {
            font-size: var(--text-sm);
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
            font-size: var(--text-sm);
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
            border: 1px solid rgba(10, 132, 255, 0.25);
            border-radius: var(--radius-lg);
        }

        .compare-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: var(--space-3);
        }

        .compare-title {
            font-size: var(--text-sm);
            color: var(--text-secondary);
        }

        .compare-exit-btn {
            padding: var(--space-1) var(--space-3) !important;
            background: var(--error) !important;
            font-size: var(--text-sm) !important;
            box-shadow: none !important;
        }

        .compare-exit-btn:hover {
            background: #ff5a50 !important;
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
            font-size: var(--text-sm);
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
            font-size: var(--text-xs);
            color: var(--text-secondary);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        /* Comparison view (result page) */
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
            font-size: var(--text-xs);
            color: var(--text-tertiary);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 500;
        }

        /* Refiner buttons */
        .refine-btn {
            font-size: var(--text-sm) !important;
            padding: var(--space-2) var(--space-3) !important;
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
</head>
<body>
    <!-- Toast Container -->
    <div id="toastContainer" class="toast-container"></div>

    <h1>🎨 Qwen Image Generator</h1>
    <p class="subtitle">Powered by Qwen-Image-2512 on your Mac <span class="connection-status"><span id="statusDot" class="status-dot checking"></span><span id="statusText">Checking...</span></span></p>

    <div class="tabs">
        <button class="tab active" onclick="showTab('generate')">✨ Generate</button>
        <button class="tab" onclick="showTab('edit')">🖌️ Edit</button>
        <button class="tab" onclick="showTab('gallery')">🖼️ Gallery</button>
        <button class="tab" onclick="showTab('settings')">⚙️ Settings</button>
    </div>

    <!-- Generate Tab -->
    <div id="tab-generate" class="tab-content active">
        <div class="input-section">
            <!-- Quick Presets -->
            <label>Quick Presets</label>
            <div class="presets">
                <span class="preset-btn active" onclick="applyPreset('quick')" data-preset="quick">🚀 Fast</span>
                <span class="preset-btn" onclick="applyPreset('quality')" data-preset="quality">✨ Quality</span>
                <span class="preset-btn" onclick="applyPreset('portrait')" data-preset="portrait">👤 Portrait</span>
                <span class="preset-btn" onclick="applyPreset('landscape')" data-preset="landscape">🏞️ Landscape</span>
            </div>

            <label for="prompt">Describe your image</label>
            <div style="position: relative;">
                <textarea id="prompt" placeholder="A majestic dragon flying over mountains at sunset..."></textarea>
                <!-- Floating AI buttons - 44px touch targets -->
                <div style="position: absolute; right: 4px; top: 4px; display: flex; flex-direction: row; gap: 4px;">
                    <button type="button" onclick="refineLocal('refine')" title="Refine" class="ai-float-btn">✨</button>
                    <button type="button" onclick="refineLocal('expand')" title="Expand" class="ai-float-btn">📝</button>
                    <button type="button" onclick="refineLocal('style')" title="Style" class="ai-float-btn">🎲</button>
                </div>
            </div>

            <div class="advanced-toggle" onclick="toggleAdvanced()">▶ Advanced Options</div>
            <div class="advanced-section" id="advancedSection">
                <!-- Prompt History -->
                <div class="option-group" style="margin-bottom: var(--space-3);">
                    <label>Prompt History</label>
                    <span class="option-hint">Click to reuse a previous prompt</span>
                    <div class="history-container" style="margin-top: var(--space-1); display: inline-block;">
                        <div class="history-btn" onclick="toggleHistory()">📜 Recent Prompts <span id="historyCount">(0)</span></div>
                        <div class="history-dropdown" id="historyDropdown"></div>
                    </div>
                </div>

                <div class="option-group" style="margin-bottom: var(--space-3);">
                    <label for="negativePrompt">Negative Prompt</label>
                    <span class="option-hint">Describe what you DON'T want in the image</span>
                    <textarea id="negativePrompt" placeholder="e.g., blurry, ugly, distorted, low quality" style="height: 60px;"></textarea>
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
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <input type="number" id="batchSize" class="batch-input" value="1" min="1" max="4" title="Generate multiple at once">
                        </div>
                    </div>
                </div>
            </div>

            <div class="options-row">
                <div class="option-group">
                    <label>Mode</label>
                    <select id="mode" onchange="updateEstimate(); clearPresetHighlight();">
                        <option value="lightning">⚡ Lightning (Fast)</option>
                        <option value="normal">🎨 Normal (Quality)</option>
                    </select>
                </div>
                <div class="option-group">
                    <label>Resolution</label>
                    <select id="resolution" onchange="updateEstimate(); clearPresetHighlight();">
                        <option value="512">512px</option>
                        <option value="768">768px</option>
                        <option value="1024">1024px</option>
                    </select>
                </div>
                <div class="option-group">
                    <label>Aspect</label>
                    <select id="aspect" onchange="clearPresetHighlight();">
                        <option value="square">◻️ Square</option>
                        <option value="landscape">▬ Landscape</option>
                        <option value="portrait">▮ Portrait</option>
                    </select>
                </div>
            </div>
            <div class="time-estimate" id="timeEstimate">⏱️ Estimated: ~1 minute</div>

            <div class="btn-row">
                <button id="generateBtn" onclick="generate()">✨ Generate</button>
                <button id="cancelBtn" class="btn-cancel" onclick="cancelGeneration()" style="display:none;">Cancel</button>
                <button class="btn-secondary" id="addToQueueBtn" onclick="addToQueue()">Queue</button>
                <button class="btn-secondary" id="regenerateBtn" onclick="regenerate()" disabled>Redo</button>
                <button class="btn-secondary" id="compareBtn" onclick="toggleSplitCompare()">Compare</button>
            </div>

            <!-- Generation Queue -->
            <div id="queueContainer" class="queue-container">
                <div class="queue-header">
                    <span class="queue-title">📋 Generation Queue</span>
                    <span class="queue-count" id="queueCount">0</span>
                </div>
                <div class="queue-list" id="queueList"></div>
                <div class="queue-actions">
                    <button class="queue-btn queue-btn-start" id="queueStartBtn" onclick="processQueue()" disabled>▶ Start Queue</button>
                    <button class="queue-btn queue-btn-clear" onclick="clearQueue()">🗑️ Clear All</button>
                </div>
            </div>
        </div>

        <!-- Split Compare Mode -->
        <div id="splitCompareMode" class="split-compare-container">
            <div class="split-compare-header">
                <span class="split-compare-title">⚔️ Split Compare Mode</span>
                <span class="split-compare-hint">Write prompts for each side, then generate one at a time</span>
                <button onclick="toggleSplitCompare()" class="compare-exit-btn">✕ Exit</button>
            </div>
            <div class="split-compare-panels">
                <div class="split-panel" id="splitPanelA">
                    <div class="split-panel-header">
                        <span class="split-panel-label">A</span>
                        <span class="split-panel-title">Left Side</span>
                    </div>
                    <textarea id="promptA" class="split-prompt" placeholder="Enter prompt for left image..."></textarea>
                    <button onclick="generateSplit('A')" class="split-generate-btn" id="generateBtnA">✨ Generate A</button>
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
                    <button onclick="generateSplit('B')" class="split-generate-btn" id="generateBtnB">✨ Generate B</button>
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
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <h3 style="margin: 0;">💡 Prompt Ideas</h3>
                <button type="button" onclick="showGenerateTemplates()" style="padding: 6px 12px; background: rgba(72, 187, 120, 0.2); border: 1px solid rgba(72, 187, 120, 0.5); border-radius: 6px; color: #fff; cursor: pointer; font-size: 12px;">📋 Templates</button>
            </div>
            <div style="display: flex; gap: 6px; flex-wrap: wrap;">
                <span class="example-btn" onclick="setPrompt('A cute robot cat in a cozy coffee shop, warm lighting, digital art')">🤖 Robot Cat</span>
                <span class="example-btn" onclick="setPrompt('Hyperrealistic portrait of a woman with freckles, natural lighting, sharp focus, professional photography')">👤 Portrait</span>
                <span class="example-btn" onclick="setPrompt('Japanese garden with cherry blossoms and wooden bridge, watercolor style')">🌸 Japanese Garden</span>
                <span class="example-btn" onclick="setPrompt('Cyberpunk city at night with neon signs and rain reflections, cinematic')">🌃 Cyberpunk</span>
                <span class="example-btn" onclick="setPrompt('Magical forest with glowing mushrooms and fireflies, fantasy art, detailed')">🍄 Magic Forest</span>
                <span class="example-btn" onclick="setPrompt('Majestic dragon flying over mountains at sunset, epic fantasy, highly detailed')">🐉 Dragon</span>
                <span class="example-btn" onclick="setPrompt('Cozy cabin in snowy mountains, warm light from windows, winter atmosphere')">🏔️ Winter Cabin</span>
                <span class="example-btn" onclick="setPrompt('Futuristic spaceship interior, sci-fi, volumetric lighting, ultra detailed')">🚀 Spaceship</span>
            </div>
        </div>
    </div>

    <!-- Edit Tab -->
    <div id="tab-edit" class="tab-content">
        <div class="input-section">
            <label>Upload Image to Edit</label>
            <div style="display: flex; gap: var(--space-2); margin-bottom: var(--space-2);">
                <button type="button" class="btn-secondary" style="flex: 1;" onclick="document.getElementById('imageUpload').click()">📁 Upload File</button>
                <button type="button" class="btn-secondary" style="flex: 1;" onclick="openGalleryPicker()">🖼️ From Gallery</button>
            </div>
            <div class="upload-area" id="uploadArea" onclick="document.getElementById('imageUpload').click()">
                <div id="uploadPlaceholder">📁 Click or drag image here</div>
                <img id="uploadPreview" class="upload-preview" style="display:none;">
                <input type="file" id="imageUpload" accept="image/*" style="display:none;" onchange="handleUpload(event)">
            </div>

            <label for="editPrompt">What changes do you want?</label>
            <textarea id="editPrompt" placeholder="e.g., Change the sky to sunset, Add a rainbow, Make it look like winter"></textarea>

            <!-- Edit Mode Selection - Simple Row of Buttons -->
            <div style="margin: 15px 0;">
                <label>Edit Mode & Tools</label>
                <div style="display: flex; gap: 8px; margin-top: 10px; flex-wrap: wrap;">
                    <button type="button" id="modeStandard" class="mode-btn active" onclick="selectEditMode('standard')" style="padding: 10px 16px; background: rgba(102, 126, 234, 0.4); border: 2px solid #667eea; border-radius: 6px; color: #fff; cursor: pointer; font-size: 13px;">🖌️ Standard</button>
                    <button type="button" id="modeAngles" class="mode-btn" onclick="selectEditMode('angles')" style="padding: 10px 16px; background: rgba(255,255,255,0.1); border: 2px solid transparent; border-radius: 6px; color: #fff; cursor: pointer; font-size: 13px;">📐 Angles</button>
                    <button type="button" id="modeUpscale" class="mode-btn" onclick="selectEditMode('upscale')" style="padding: 10px 16px; background: rgba(255,255,255,0.1); border: 2px solid transparent; border-radius: 6px; color: #fff; cursor: pointer; font-size: 13px;">🔍 Upscale</button>
                    <span style="border-left: 1px solid rgba(255,255,255,0.2); margin: 0 4px;"></span>
                    <button type="button" onclick="showEditCheatsheet()" style="padding: 10px 16px; background: rgba(72, 187, 120, 0.2); border: 1px solid rgba(72, 187, 120, 0.5); border-radius: 6px; color: #fff; cursor: pointer; font-size: 13px;">📋 Templates</button>
                    <button type="button" id="angleCheatBtn" onclick="showAngleCheatsheet()" style="padding: 10px 16px; background: rgba(237, 137, 54, 0.2); border: 1px solid rgba(237, 137, 54, 0.5); border-radius: 6px; color: #fff; cursor: pointer; font-size: 13px; display: none;">📐 Angle Guide</button>
                </div>
                <!-- Upscale resolution options (shown when upscale mode selected) -->
                <div id="upscaleControls" style="display: none; margin-top: 12px;">
                    <div style="display: flex; gap: 10px;">
                        <button type="button" class="upscale-btn active" data-res="2K" onclick="selectUpscale('2K')" style="flex: 1; padding: 12px; text-align: center; background: rgba(102, 126, 234, 0.4); border: 2px solid #667eea; border-radius: 6px; cursor: pointer; color: #fff;">
                            <strong>2K</strong> <span style="font-size: 11px; opacity: 0.7;">~2048px</span>
                        </button>
                        <button type="button" class="upscale-btn" data-res="4K" onclick="selectUpscale('4K')" style="flex: 1; padding: 12px; text-align: center; background: rgba(255,255,255,0.1); border: 2px solid transparent; border-radius: 6px; cursor: pointer; color: #fff;">
                            <strong>4K</strong> <span style="font-size: 11px; opacity: 0.7;">~4096px</span>
                        </button>
                    </div>
                    <input type="hidden" id="upscaleRes" value="2K">
                </div>
                <!-- Hidden inputs for compatibility -->
                <input type="hidden" id="angleDirection" value="front">
                <input type="hidden" id="angleElevation" value="eye">
                <input type="hidden" id="angleDistance" value="medium">
            </div>

            <button onclick="editImage()" id="editBtn">🖌️ Apply Edit</button>
        </div>
        <div id="editResult"></div>
    </div>

    <!-- Gallery Tab -->
    <div id="tab-gallery" class="tab-content">
        <div class="filter-tabs">
            <div class="filter-tab active" onclick="filterGallery('all')">All <span id="galleryCount" class="gallery-count"></span></div>
            <div class="filter-tab" onclick="filterGallery('recent')">🕐 Recent</div>
            <div class="filter-tab" onclick="filterGallery('favorites')">⭐ Favorites</div>
            <div class="filter-tab" onclick="filterGallery('lightning')">⚡ Lightning</div>
            <div class="filter-tab" onclick="filterGallery('normal')">🎨 Normal</div>
            <div class="filter-tab" onclick="filterGallery('edit')">🖌️ Edit</div>
            <div class="filter-tab" onclick="enterCompareMode()">⚔️ Compare</div>
        </div>
        <div id="compareMode" class="compare-mode-container">
            <div class="compare-header">
                <span class="compare-title">📊 Compare Mode: Select 2 images to compare side-by-side</span>
                <button onclick="exitCompareMode()" class="compare-exit-btn">✕ Exit</button>
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
            <h2 style="margin-top: 0;">⚙️ Settings</h2>

            <!-- Provider Toggle -->
            <div style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 12px; margin-bottom: 20px;">
                <label style="font-size: 1.1em; margin-bottom: 15px; display: block;">Image Generation</label>
                <div style="background: rgba(72, 187, 120, 0.1); padding: 15px; border-radius: 12px; border: 1px solid rgba(72, 187, 120, 0.3);">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="font-size: 1.5em;">💻</span>
                        <div>
                            <div style="font-weight: bold;">Local Mode (ComfyUI + Qwen)</div>
                            <div style="font-size: 0.85em; color: #888;">Image generation on your Mac</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Local Models Info -->
            <div style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 12px; margin-bottom: 20px;">
                <label>Local Models</label>
                <div style="margin-top: 10px; font-size: 0.9em; color: #aaa;">
                    <div style="margin-bottom: 8px;">🖼️ <strong>Image Gen:</strong> Qwen-Image-2512 Q6_K + Abliterated Text Encoder Q6_K</div>
                    <div style="margin-bottom: 8px;">🖌️ <strong>Image Edit:</strong> Qwen-Image-Edit-2511 Q4_K_M (balanced VRAM)</div>
                    <div>✨ <strong>Refinement:</strong> Qwen 2.5 0.5B via Ollama (~350MB)</div>
                </div>
            </div>

            <!-- Status -->
            <div id="settingsStatus" style="padding: 15px; background: rgba(72, 187, 120, 0.2); border-radius: 8px; display: none; margin-top: 15px;">
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
    <div class="modal" id="galleryPickerModal" onclick="closeGalleryPicker(event)" style="display: none;">
        <div onclick="event.stopPropagation()" style="background: rgba(30, 30, 50, 0.98); backdrop-filter: blur(20px); padding: var(--space-4); border-radius: var(--radius-lg); max-width: 600px; max-height: 80vh; overflow-y: auto;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: var(--space-3);">
                <h3 style="margin: 0;">Select Image to Edit</h3>
                <button onclick="closeGalleryPicker()" style="background: none; border: none; color: var(--text-tertiary); font-size: 20px; cursor: pointer;">&times;</button>
            </div>
            <div id="galleryPickerGrid" class="gallery" style="grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));"></div>
        </div>
    </div>

    <!-- Angle Cheatsheet Modal -->
    <div class="modal" id="angleCheatsheetModal" onclick="closeAngleCheatsheet(event)" style="display: none;">
        <div onclick="event.stopPropagation()" style="background: rgba(30, 30, 50, 0.98); backdrop-filter: blur(20px); padding: var(--space-4); border-radius: var(--radius-lg); max-width: 650px; max-height: 85vh; overflow-y: auto;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: var(--space-3);">
                <h3 style="margin: 0;">📐 Camera Angle Prompt Cheatsheet</h3>
                <button onclick="closeAngleCheatsheet()" style="background: none; border: none; color: var(--text-tertiary); font-size: 20px; cursor: pointer;">&times;</button>
            </div>

            <div style="font-size: 13px; color: var(--text-secondary); margin-bottom: var(--space-3);">
                <p style="margin: 0 0 10px 0;"><strong>Format:</strong> <code style="background: rgba(102,126,234,0.2); padding: 2px 6px; border-radius: 4px;">&lt;sks&gt; [direction] [elevation] [distance]</code></p>
                <p style="margin: 0; color: #aaa;">Copy the angle prompt and paste it into the "Edit Description" field above.</p>
            </div>

            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: var(--space-3);">
                <!-- Directions -->
                <div style="background: rgba(255,255,255,0.05); padding: var(--space-2); border-radius: var(--radius-md);">
                    <h4 style="margin: 0 0 10px 0; color: var(--accent); font-size: 13px;">🧭 Direction</h4>
                    <div style="font-size: 12px; line-height: 1.8;">
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

                <!-- Elevation -->
                <div style="background: rgba(255,255,255,0.05); padding: var(--space-2); border-radius: var(--radius-md);">
                    <h4 style="margin: 0 0 10px 0; color: var(--accent); font-size: 13px;">📏 Elevation</h4>
                    <div style="font-size: 12px; line-height: 1.8;">
                        <div><code>overhead</code> - Bird's eye</div>
                        <div><code>high</code> - Above eye</div>
                        <div><code>eye</code> - Eye level</div>
                        <div><code>low</code> - Below eye</div>
                        <div><code>ground</code> - Ground level</div>
                    </div>
                </div>

                <!-- Distance -->
                <div style="background: rgba(255,255,255,0.05); padding: var(--space-2); border-radius: var(--radius-md);">
                    <h4 style="margin: 0 0 10px 0; color: var(--accent); font-size: 13px;">🔍 Distance</h4>
                    <div style="font-size: 12px; line-height: 1.8;">
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

            <div style="margin-top: var(--space-3); padding: var(--space-2); background: rgba(102,126,234,0.1); border-radius: var(--radius-md); border: 1px solid rgba(102,126,234,0.3);">
                <h4 style="margin: 0 0 10px 0; font-size: 13px;">✨ Examples (click to copy)</h4>
                <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                    <button onclick="copyAnglePrompt('&lt;sks&gt; right eye medium')" style="padding: 6px 12px; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); border-radius: 4px; color: #fff; cursor: pointer; font-size: 12px;">Right profile</button>
                    <button onclick="copyAnglePrompt('&lt;sks&gt; left eye medium')" style="padding: 6px 12px; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); border-radius: 4px; color: #fff; cursor: pointer; font-size: 12px;">Left profile</button>
                    <button onclick="copyAnglePrompt('&lt;sks&gt; front high close')" style="padding: 6px 12px; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); border-radius: 4px; color: #fff; cursor: pointer; font-size: 12px;">High angle close</button>
                    <button onclick="copyAnglePrompt('&lt;sks&gt; front low full')" style="padding: 6px 12px; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); border-radius: 4px; color: #fff; cursor: pointer; font-size: 12px;">Low full body</button>
                    <button onclick="copyAnglePrompt('&lt;sks&gt; back_left eye medium')" style="padding: 6px 12px; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); border-radius: 4px; color: #fff; cursor: pointer; font-size: 12px;">Over shoulder</button>
                    <button onclick="copyAnglePrompt('&lt;sks&gt; front overhead wide')" style="padding: 6px 12px; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); border-radius: 4px; color: #fff; cursor: pointer; font-size: 12px;">Bird's eye wide</button>
                </div>
            </div>

            <div style="margin-top: var(--space-2); text-align: center;">
                <button onclick="closeAngleCheatsheet()" style="padding: 10px 24px; background: var(--accent); border: none; border-radius: var(--radius-md); color: #fff; cursor: pointer; font-size: 14px;">Got it!</button>
            </div>
        </div>
    </div>

    <!-- Edit Prompt Template Modal -->
    <div class="modal" id="editCheatsheetModal" onclick="closeEditCheatsheet(event)" style="display: none;">
        <div onclick="event.stopPropagation()" style="background: rgba(30, 30, 50, 0.98); backdrop-filter: blur(20px); padding: var(--space-4); border-radius: var(--radius-lg); max-width: 500px; max-height: 85vh; overflow-y: auto;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: var(--space-3);">
                <h3 style="margin: 0;">🖌️ Edit Templates</h3>
                <button onclick="closeEditCheatsheet()" style="background: none; border: none; color: var(--text-tertiary); font-size: 20px; cursor: pointer;">&times;</button>
            </div>

            <div style="font-size: 13px; color: var(--text-secondary); margin-bottom: var(--space-3);">
                <p style="margin: 0;">Select a template and fill in keywords, then click "Use Template"</p>
            </div>

            <!-- Template Selection -->
            <div style="margin-bottom: var(--space-3);">
                <label style="font-size: 12px; color: var(--text-secondary); margin-bottom: 6px; display: block;">Template</label>
                <select id="editTemplateSelect" onchange="updateTemplatePreview()" style="width: 100%; padding: 10px 12px; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); border-radius: 6px; color: #fff; font-size: 14px; cursor: pointer;">
                    <optgroup label="🌄 Background">
                        <option value="Change the background to {location}">Change background to...</option>
                        <option value="Replace background with {scene}">Replace background with...</option>
                        <option value="Add {weather} weather">Add weather...</option>
                    </optgroup>
                    <optgroup label="💡 Lighting">
                        <option value="Add {type} lighting">Add lighting...</option>
                        <option value="Change lighting to {time} atmosphere">Change to time of day...</option>
                        <option value="Add {color} colored lighting">Add colored lighting...</option>
                    </optgroup>
                    <optgroup label="🎨 Style">
                        <option value="Make it look like {style}">Make it look like...</option>
                        <option value="Convert to {art_style} style">Convert to art style...</option>
                        <option value="Apply {effect} effect">Apply effect...</option>
                    </optgroup>
                    <optgroup label="✨ Add Elements">
                        <option value="Add {objects} in the scene">Add objects...</option>
                        <option value="Add falling {particles}">Add falling particles...</option>
                        <option value="Add {effect} effects">Add visual effects...</option>
                    </optgroup>
                    <optgroup label="👤 Portrait">
                        <option value="Change hair color to {color}">Change hair color...</option>
                        <option value="Add {accessory}">Add accessory...</option>
                        <option value="Change outfit to {clothing}">Change outfit...</option>
                        <option value="Change expression to {expression}">Change expression...</option>
                    </optgroup>
                    <optgroup label="🎭 Mood">
                        <option value="Make it more {mood}">Change mood to...</option>
                        <option value="Add {atmosphere} atmosphere">Add atmosphere...</option>
                    </optgroup>
                </select>
            </div>

            <!-- Keyword Input -->
            <div style="margin-bottom: var(--space-3);">
                <label style="font-size: 12px; color: var(--text-secondary); margin-bottom: 6px; display: block;">Your Keyword</label>
                <input type="text" id="editTemplateKeyword" placeholder="e.g., sunset beach, neon pink, anime..."
                       style="width: 100%; padding: 10px 12px; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); border-radius: 6px; color: #fff; font-size: 14px; box-sizing: border-box;"
                       onkeyup="updateTemplatePreview()" onkeypress="if(event.key==='Enter')applyTemplate()">
            </div>

            <!-- Preview -->
            <div style="margin-bottom: var(--space-3); padding: var(--space-2); background: rgba(102, 126, 234, 0.1); border: 1px solid rgba(102, 126, 234, 0.3); border-radius: 6px;">
                <label style="font-size: 11px; color: var(--text-tertiary); margin-bottom: 4px; display: block;">Preview</label>
                <div id="templatePreview" style="font-size: 14px; color: #fff; min-height: 20px;">Select a template and enter a keyword</div>
            </div>

            <!-- Quick Keywords -->
            <div style="margin-bottom: var(--space-3);">
                <label style="font-size: 12px; color: var(--text-secondary); margin-bottom: 8px; display: block;">Quick Keywords (click to use)</label>
                <div id="quickKeywords" style="display: flex; flex-wrap: wrap; gap: 6px;">
                    <!-- Populated by JS based on template -->
                </div>
            </div>

            <!-- Action Buttons -->
            <div style="display: flex; gap: 10px;">
                <button onclick="applyTemplate()" style="flex: 1; padding: 12px; background: var(--accent); border: none; border-radius: var(--radius-md); color: #fff; cursor: pointer; font-size: 14px; font-weight: 500;">✓ Use Template</button>
                <button onclick="closeEditCheatsheet()" style="padding: 12px 20px; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); border-radius: var(--radius-md); color: #fff; cursor: pointer; font-size: 14px;">Cancel</button>
            </div>
        </div>
    </div>

    <!-- Generate Templates Modal -->
    <div class="modal" id="generateTemplatesModal" onclick="closeGenerateTemplates(event)" style="display: none;">
        <div onclick="event.stopPropagation()" style="background: rgba(30, 30, 50, 0.98); backdrop-filter: blur(20px); padding: var(--space-4); border-radius: var(--radius-lg); max-width: 500px; max-height: 85vh; overflow-y: auto;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: var(--space-3);">
                <h3 style="margin: 0;">✨ Generate Templates</h3>
                <button onclick="closeGenerateTemplates()" style="background: none; border: none; color: var(--text-tertiary); font-size: 20px; cursor: pointer;">&times;</button>
            </div>

            <div style="font-size: 13px; color: var(--text-secondary); margin-bottom: var(--space-3);">
                <p style="margin: 0;">Select a template, fill in your subject, and generate!</p>
            </div>

            <!-- Template Selection -->
            <div style="margin-bottom: var(--space-3);">
                <label style="font-size: 12px; color: var(--text-secondary); margin-bottom: 6px; display: block;">Template</label>
                <select id="genTemplateSelect" onchange="updateGenTemplatePreview()" style="width: 100%; padding: 10px 12px; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); border-radius: 6px; color: #fff; font-size: 14px; cursor: pointer;">
                    <optgroup label="👤 People">
                        <option value="Hyperrealistic portrait of {subject}, natural lighting, sharp focus, professional photography">Realistic Portrait</option>
                        <option value="Anime style illustration of {subject}, vibrant colors, detailed">Anime Character</option>
                        <option value="Fantasy warrior {subject}, epic armor, dramatic lighting, digital art">Fantasy Warrior</option>
                    </optgroup>
                    <optgroup label="🏞️ Landscapes">
                        <option value="{subject} landscape at sunset, golden hour, cinematic, breathtaking">Sunset Landscape</option>
                        <option value="Magical {subject} with glowing elements, fantasy art, ethereal atmosphere">Fantasy Scene</option>
                        <option value="{subject} in winter with snow, cozy atmosphere, warm lights">Winter Scene</option>
                    </optgroup>
                    <optgroup label="🎨 Art Styles">
                        <option value="{subject}, oil painting style, classical art, museum quality">Oil Painting</option>
                        <option value="{subject}, watercolor style, soft colors, artistic">Watercolor</option>
                        <option value="{subject}, Studio Ghibli style, whimsical, animated">Studio Ghibli</option>
                        <option value="{subject}, cyberpunk aesthetic, neon lights, futuristic">Cyberpunk</option>
                    </optgroup>
                    <optgroup label="🐾 Animals & Creatures">
                        <option value="Cute {subject}, adorable, fluffy, heartwarming, detailed fur">Cute Animal</option>
                        <option value="Majestic {subject}, powerful, detailed, nature photography style">Majestic Animal</option>
                        <option value="Mythical {subject} creature, fantasy art, magical, highly detailed">Fantasy Creature</option>
                    </optgroup>
                    <optgroup label="🏛️ Architecture & Objects">
                        <option value="{subject} interior, cozy atmosphere, warm lighting, detailed">Cozy Interior</option>
                        <option value="Futuristic {subject}, sci-fi design, sleek, volumetric lighting">Sci-Fi Design</option>
                        <option value="Ancient {subject}, mysterious, dramatic lighting, epic scale">Ancient/Epic</option>
                    </optgroup>
                </select>
            </div>

            <!-- Subject Input -->
            <div style="margin-bottom: var(--space-3);">
                <label style="font-size: 12px; color: var(--text-secondary); margin-bottom: 6px; display: block;">Your Subject</label>
                <input type="text" id="genTemplateSubject" placeholder="e.g., a woman with red hair, mountain range, dragon..."
                       style="width: 100%; padding: 10px 12px; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); border-radius: 6px; color: #fff; font-size: 14px; box-sizing: border-box;"
                       onkeyup="updateGenTemplatePreview()" onkeypress="if(event.key==='Enter')applyGenTemplate()">
            </div>

            <!-- Preview -->
            <div style="margin-bottom: var(--space-3); padding: var(--space-2); background: rgba(102, 126, 234, 0.1); border: 1px solid rgba(102, 126, 234, 0.3); border-radius: 6px;">
                <label style="font-size: 11px; color: var(--text-tertiary); margin-bottom: 4px; display: block;">Preview</label>
                <div id="genTemplatePreview" style="font-size: 13px; color: #fff; min-height: 40px; line-height: 1.4;">Select a template and enter your subject</div>
            </div>

            <!-- Quick Subjects -->
            <div style="margin-bottom: var(--space-3);">
                <label style="font-size: 12px; color: var(--text-secondary); margin-bottom: 8px; display: block;">Quick Subjects (click to use)</label>
                <div id="genQuickSubjects" style="display: flex; flex-wrap: wrap; gap: 6px;">
                    <!-- Populated by JS -->
                </div>
            </div>

            <!-- Action Buttons -->
            <div style="display: flex; gap: 10px;">
                <button onclick="applyGenTemplate()" style="flex: 1; padding: 12px; background: var(--accent); border: none; border-radius: var(--radius-md); color: #fff; cursor: pointer; font-size: 14px; font-weight: 500;">✓ Use Template</button>
                <button onclick="closeGenerateTemplates()" style="padding: 12px 20px; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); border-radius: var(--radius-md); color: #fff; cursor: pointer; font-size: 14px;">Cancel</button>
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
                success: '✅',
                error: '❌',
                warning: '⚠️',
                info: 'ℹ️'
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
            startBtn.textContent = queueProcessing ? '⏳ Processing...' : '▶ Start Queue';

            list.innerHTML = generationQueue.map((item, idx) => {
                const statusIcon = item.status === 'processing' ? '⏳' : item.status === 'done' ? '✅' : item.status === 'error' ? '❌' : '⏸️';
                const modeIcon = item.mode === 'lightning' ? '⚡' : '🎨';
                return '<div class="queue-item' + (item.status === 'processing' ? ' processing' : '') + '">' +
                    '<span class="queue-item-status">' + statusIcon + '</span>' +
                    '<span class="queue-item-prompt" title="' + item.prompt.replace(/"/g, '&quot;') + '">' + item.prompt + '</span>' +
                    '<span class="queue-item-settings">' + modeIcon + ' ' + item.resolution + 'px</span>' +
                    (item.status === 'pending' ? '<span class="queue-item-remove" onclick="removeFromQueue(' + item.id + ')">✕</span>' : '') +
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
                btn.textContent = '⏳ Queue ' + (i + 1) + '/' + generationQueue.length;
                status.style.display = 'block';
                status.className = 'generating';
                statusText.textContent = '📋 Processing queue item ' + (i + 1) + '/' + generationQueue.length;

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
                            scheduler: item.scheduler
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
                            '<a href="' + data.image + '" download="' + filename + '"><button class="btn-green">⬇️ Download</button></a>' +
                            '<button onclick="toggleFavorite(\\'' + filename + '\\')">⭐ Favorite</button>' +
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
            btn.textContent = '✨ Generate';
            status.className = 'success';
            statusText.textContent = '✅ Queue complete!';
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
            toggle.textContent = section.classList.contains('show') ? '▼ Advanced Options' : '▶ Advanced Options';
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
                    return '<div class="history-item" style="display:flex; justify-content:space-between; align-items:center;">' +
                        '<div onclick="useHistoryPrompt(' + originalIndex + ')" style="flex:1; cursor:pointer;">' +
                        '<div>' + item.prompt.substring(0, 60) + (item.prompt.length > 60 ? '...' : '') + '</div>' +
                        '<div class="history-meta">' + item.mode + ' | ' + item.resolution + 'px' +
                        (item.timestamp ? ' | ' + new Date(item.timestamp).toLocaleDateString() : '') + '</div>' +
                        '</div>' +
                        '<span onclick="event.stopPropagation(); deleteHistoryItem(' + originalIndex + ')" style="cursor:pointer; padding:5px; opacity:0.6;" title="Delete">🗑️</span>' +
                        '</div>';
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
            const times = { lightning: { 512: 1, 768: 2, 1024: 3 }, normal: { 512: 7, 768: 12, 1024: 16 } };
            const totalTime = times[mode][resolution] * batch;
            document.getElementById('timeEstimate').textContent = '⏱️ Estimated: ~' + totalTime + ' min' + (batch > 1 ? ' (for ' + batch + ' images)' : '');
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
                    statusText.textContent = '📦 ' + data.message;
                    progressContainer.style.display = 'none';
                } else if (data.status === 'generating') {
                    progressContainer.style.display = 'block';
                    const percent = Math.round((data.current_step / data.total_steps) * 100);
                    progressFill.style.width = percent + '%';
                    stepText.textContent = 'Step ' + data.current_step + '/' + data.total_steps;
                    percentText.textContent = percent + '%';
                    statusText.textContent = '🎨 Generating...';
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
                        ? '📦 Image ' + (currentBatchIndex + 1) + '/' + totalBatchSize + ': Loading models...'
                        : '📦 ' + data.message;
                    progressContainer.style.display = 'none';
                } else if (data.status === 'generating') {
                    progressContainer.style.display = 'block';
                    const percent = Math.round((data.current_step / data.total_steps) * 100);
                    progressFill.style.width = percent + '%';
                    stepText.textContent = 'Step ' + data.current_step + '/' + data.total_steps;
                    percentText.textContent = percent + '%';
                    statusText.textContent = totalBatchSize > 1
                        ? '🎨 Image ' + (currentBatchIndex + 1) + '/' + totalBatchSize + ' generating...'
                        : '🎨 Generating...';
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
                let icon = '⏳';
                if (completedIndices.includes(i)) {
                    className += ' done';
                    icon = '✅';
                } else if (i === currentIndex) {
                    className += ' active';
                    icon = '🎨';
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
                    btn.textContent = batchSize > 1 ? '⏳ ' + (i + 1) + '/' + batchSize : '⏳ Generating...';
                    statusText.textContent = batchSize > 1 ? '🚀 Starting image ' + (i + 1) + '...' : '🚀 Starting...';

                    // Update batch progress UI
                    if (batchSize > 1) {
                        result.innerHTML = renderBatchProgress(batchSize, i, completedIndices);
                    }

                    const queueResponse = await fetch('/queue', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({prompt, mode, resolution, aspect, negativePrompt, seed, sampler, scheduler})
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
                statusText.textContent = batchSize > 1 ? '✅ ' + batchSize + ' images done!' : '✅ Done!';
                progressContainer.style.display = 'none';

                if (batchSize === 1) {
                    const filename = generatedImages[0].split('/').pop();
                    result.innerHTML = '<img src="' + generatedImages[0] + '?t=' + Date.now() + '">' +
                        '<div class="result-actions">' +
                        '<a href="' + generatedImages[0] + '" download="' + filename + '"><button class="btn-green">⬇️ Download</button></a>' +
                        '<button onclick="toggleFavorite(\\'' + filename + '\\')">⭐ Favorite</button>' +
                        '<button class="btn-secondary" onclick="copySeed()">📋 Copy Seed</button>' +
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
                            '<a href="' + img + '" download="' + filename + '"><button class="btn-green btn-sm">⬇️</button></a>' +
                            '<button class="btn-sm" onclick="toggleFavorite(\\'' + filename + '\\')">⭐</button>' +
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
                statusText.textContent = '❌ ' + e.message;
                showToast('Error', e.message, 'error');
            } finally {
                currentPromptId = null;
                totalBatchSize = 1;
                currentBatchIndex = 0;
                btn.disabled = false;
                btn.textContent = '✨ Generate';
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
                document.getElementById('statusText').textContent = '⏹️ Cancelled';
                document.getElementById('generateBtn').disabled = false;
                document.getElementById('generateBtn').textContent = '✨ Generate';
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
            btn.textContent = '⏳ Generating...';
            resultDiv.innerHTML = '<div class="split-placeholder">Generating...</div>';

            try {
                const queueResponse = await fetch('/queue', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({prompt, mode, resolution, aspect, negativePrompt})
                });
                const queueData = await queueResponse.json();
                if (!queueData.prompt_id) throw new Error(queueData.error || 'Failed to queue');

                const response = await fetch('/wait?prompt_id=' + queueData.prompt_id);
                const data = await response.json();

                if (data.success) {
                    resultDiv.innerHTML = '<img src="' + data.image + '?t=' + Date.now() + '">';
                } else {
                    resultDiv.innerHTML = '<div class="split-placeholder" style="color: var(--error);">❌ ' + data.error + '</div>';
                }
            } catch (e) {
                resultDiv.innerHTML = '<div class="split-placeholder" style="color: var(--error);">❌ ' + e.message + '</div>';
            } finally {
                // Always reset button state
                btn.disabled = false;
                btn.textContent = '✨ Generate ' + side;
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
            btn.textContent = '⏳ Comparing...';
            status.style.display = 'block';
            status.className = 'generating';
            statusText.textContent = '⚡ Generating Lightning version...';
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

                statusText.textContent = '🎨 Generating Normal version (this takes longer)...';

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
                statusText.textContent = '✅ Comparison complete!';

                // Show side-by-side comparison
                result.innerHTML =
                    '<h3 style="text-align:center;margin-bottom:15px;">⚔️ Lightning vs Normal (Same Seed: ' + seed + ')</h3>' +
                    '<div class="comparison">' +
                    '<div class="comparison-col">' +
                    '<div class="comparison-label">⚡ Lightning (4 steps, ~1 min)</div>' +
                    (lightningData.success ? '<img src="' + lightningData.image + '?t=' + Date.now() + '">' : '<p>Failed</p>') +
                    '</div>' +
                    '<div class="comparison-col">' +
                    '<div class="comparison-label">🎨 Normal (30 steps, ~7 min)</div>' +
                    (normalData.success ? '<img src="' + normalData.image + '?t=' + Date.now() + '">' : '<p>Failed</p>') +
                    '</div>' +
                    '</div>';

                addToHistory(prompt, 'compare', resolution, aspect, negativePrompt);
            } catch (e) {
                status.className = 'error';
                statusText.textContent = '❌ ' + e.message;
            } finally {
                // Always reset button state
                btn.disabled = false;
                btn.textContent = '⚔️ Compare';
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
                    return '<div class="gallery-item" data-type="' + type + '" data-filename="' + img + '" data-timestamp="' + item.timestamp + '">' +
                        '<img src="/output/' + img + '" onclick="handleGalleryClick(' + q + img + q + ')">' +
                        '<div class="gallery-actions">' +
                        '<span class="favorite-star" onclick="event.stopPropagation(); toggleFavorite(' + q + img + q + ')">' + (favorites.includes(img) ? '⭐' : '☆') + '</span>' +
                        '<span class="delete-btn" onclick="event.stopPropagation(); deleteImage(' + q + img + q + ')">🗑️</span>' +
                        '</div>' +
                        '<div class="gallery-type-badge">' + (type === 'lightning' ? '⚡' : type === 'edit' ? '🖌️' : '🎨') + '</div>' +
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
                    'favorites': 'No favorites yet. Click ⭐ to add some!',
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

            // Update visual selection for mode buttons
            document.querySelectorAll('.mode-btn').forEach(el => {
                el.style.background = 'rgba(255,255,255,0.1)';
                el.style.borderColor = 'transparent';
            });
            const selected = document.getElementById('mode' + mode.charAt(0).toUpperCase() + mode.slice(1));
            if (selected) {
                selected.style.background = 'rgba(102, 126, 234, 0.4)';
                selected.style.borderColor = '#667eea';
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
            'color': ['blonde', 'red', 'blue', 'pink', 'silver', 'black', 'purple'],
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
            const placeholderMatch = template.match(/\{(\w+)\}/);
            const placeholderName = placeholderMatch ? placeholderMatch[1] : '';

            // Update preview
            if (keyword) {
                preview.textContent = template.replace(/\{[^}]+\}/, keyword);
                preview.style.color = '#fff';
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

            const finalPrompt = template.replace(/\{[^}]+\}/, keyword);
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
                preview.textContent = template.replace(/\{subject\}/g, subject);
                preview.style.color = '#fff';
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

            const finalPrompt = template.replace(/\{subject\}/g, subject);
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
                btn.style.background = 'rgba(255,255,255,0.1)';
                btn.style.border = 'none';
            });

            // Add active styling to clicked
            element.style.background = 'rgba(102, 126, 234, 0.4)';
            element.style.border = '2px solid #667eea';

            // Update hidden value
            const value = element.dataset.dir || element.dataset.elev || element.dataset.dist;
            document.getElementById(valueField).value = value;
        }

        // Upscale resolution selection
        function selectUpscale(resolution) {
            // Update button styling
            document.querySelectorAll('.upscale-btn').forEach(btn => {
                if (btn.dataset.res === resolution) {
                    btn.style.background = 'rgba(102, 126, 234, 0.4)';
                    btn.style.border = '2px solid #667eea';
                } else {
                    btn.style.background = 'rgba(255,255,255,0.1)';
                    btn.style.border = '2px solid transparent';
                }
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
            document.getElementById('editBtn').textContent = '⏳ Processing...';
            document.getElementById('editResult').innerHTML =
                '<div style="text-align:center; padding: var(--space-4);">' +
                '<div class="spinner" style="width:32px;height:32px;border:3px solid var(--glass-border);border-top-color:var(--accent);border-radius:50%;animation:spin 1s linear infinite;margin:0 auto;"></div>' +
                '<p style="margin-top:var(--space-2);color:var(--text-secondary);">🎨 Editing image... This may take 2-5 minutes</p>' +
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
                        '<div style="display:grid; grid-template-columns: 1fr 1fr; gap: var(--space-3); margin-bottom: var(--space-3);">' +
                        '<div style="text-align:center;">' +
                        '<div style="font-size:var(--text-sm);color:var(--text-tertiary);margin-bottom:var(--space-1);">Before</div>' +
                        '<img src="' + originalImage + '" style="width:100%;border-radius:var(--radius-md);border:1px solid var(--glass-border);">' +
                        '</div>' +
                        '<div style="text-align:center;">' +
                        '<div style="font-size:var(--text-sm);color:var(--text-tertiary);margin-bottom:var(--space-1);">After</div>' +
                        '<img src="' + data.image + '?t=' + Date.now() + '" style="width:100%;border-radius:var(--radius-md);border:1px solid var(--accent);">' +
                        '</div>' +
                        '</div>' +
                        '<div class="result-actions">' +
                        '<a href="' + data.image + '" download="' + filename + '"><button class="btn-green">⬇️ Download</button></a>' +
                        '<button onclick="toggleFavorite(\\'' + filename + '\\')">⭐ Favorite</button>' +
                        '</div>';
                    showToast('Edit Complete', 'Your image has been edited', 'success');
                } else {
                    document.getElementById('editResult').innerHTML = '<p style="color:#f56565;">❌ ' + data.error + '</p>';
                    showToast('Edit Failed', data.error, 'error');
                }
            } catch (e) {
                document.getElementById('editResult').innerHTML = '<p style="color:#f56565;">❌ ' + e.message + '</p>';
                showToast('Error', e.message, 'error');
            } finally {
                // Always reset button state, even if request times out or fails
                document.getElementById('editBtn').disabled = false;
                document.getElementById('editBtn').textContent = '🖌️ Apply Edit';
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
                self.send_header('Content-type', 'image/png')
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
            result = queue_prompt(
                data.get('prompt', ''),
                data.get('mode', 'lightning'),
                data.get('resolution', 512),
                data.get('aspect', 'square'),
                data.get('seed'),
                data.get('negativePrompt', ''),
                data.get('sampler', 'euler'),
                data.get('scheduler', 'normal')
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
        except:
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
    except:
        pass
    return defaults

def save_settings(settings):
    """Save user settings to file"""
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
        return True
    except:
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
            print(f"Unloaded Ollama model {model} to free VRAM")
        except Exception as e:
            print(f"Could not unload Ollama: {e}")

def queue_prompt(prompt, mode='lightning', resolution=512, aspect='square', seed=None, negative_prompt='', sampler='euler', scheduler='normal'):
    # Free up VRAM by unloading Ollama model before image generation
    unload_ollama_model()

    try:
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
            progress_state[prompt_id] = {'start_time': time.time(), 'mode': mode}
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
            except:
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
    except:
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
    except:
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
        except:
            pass
        return {"success": True, "history": prompt_history}
    except Exception as e:
        return {"success": False, "error": str(e)}

def save_favorites(favorites):
    try:
        with open(FAVORITES_FILE, 'w') as f:
            json.dump(favorites, f)
    except:
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
    except:
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
            from urllib.parse import urlencode

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
            except:
                pass

        return {"success": False, "error": "Edit timeout"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def check_comfyui():
    try:
        urllib.request.urlopen(f"{COMFYUI_URL}/system_stats", timeout=2)
        return True
    except:
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
    except:
        return "localhost"

def main():
    print("=" * 50)
    print("  🎨 Qwen Image Generator - Enhanced")
    print("=" * 50)
    print()

    if not check_comfyui():
        print("Starting ComfyUI backend...")
        if not start_comfyui():
            print("❌ Failed to start ComfyUI. Please run it manually.")
            sys.exit(1)

    local_ip = get_local_ip()
    print("✅ ComfyUI backend running")
    print()
    print("🌐 Access the generator:")
    print(f"   Local:   http://localhost:8080")
    print(f"   Network: http://{local_ip}:8080")
    print()
    print("Press Ctrl+C to stop")
    print()

    threading.Timer(1.5, lambda: webbrowser.open('http://localhost:8080')).start()

    server = HTTPServer(('0.0.0.0', 8080), RequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        server.shutdown()

if __name__ == "__main__":
    main()
