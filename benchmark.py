#!/usr/bin/env python3
"""
Qwen Image Generator Benchmark Script

Compares generation times across different models, resolutions, and modes.
Run with: python benchmark.py

Usage:
  python benchmark.py                    # Run all benchmarks
  python benchmark.py --quick            # Quick test (512x512 only)
  python benchmark.py --model Q6_K       # Test specific model
  python benchmark.py --compare          # Compare Q4 vs Q6 (if both available)
"""

import requests
import json
import time
import argparse
import os
from datetime import datetime

COMFYUI_URL = "http://127.0.0.1:8188"
OUTPUT_DIR = "./output"

# Test configurations
RESOLUTIONS = [
    ("512x512", 512, 512),
    ("768x768", 768, 768),
    ("1024x1024", 1024, 1024),
]

QUICK_RESOLUTIONS = [
    ("512x512", 512, 512),
]

TEST_PROMPT = "A red apple on a white table, studio lighting, professional photography"

def check_comfyui():
    """Check if ComfyUI is running"""
    try:
        r = requests.get(f"{COMFYUI_URL}/system_stats", timeout=5)
        return r.status_code == 200
    except:
        return False

def get_current_model():
    """Get the currently configured model from workflow"""
    # This would need to read from the app config
    # For now, check what's in models/unet
    unet_dir = "./models/unet"
    if os.path.exists(unet_dir):
        models = [f for f in os.listdir(unet_dir) if f.endswith('.gguf') and 'edit' not in f.lower()]
        return models
    return []

def build_workflow(width, height, prompt, lightning=True, steps=4, cfg=1.0, seed=-1):
    """Build a ComfyUI workflow for benchmarking"""
    if seed == -1:
        import random
        seed = random.randint(0, 2**32 - 1)

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
                "text": prompt,
                "clip": ["3", 0]
            }
        },
        "5": {
            "class_type": "UnetLoaderGGUF",
            "inputs": {
                "unet_name": "qwen-image-Q6_K.gguf"  # Will be replaced
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
                "filename_prefix": "benchmark",
                "images": ["10", 0]
            }
        }
    }

    if lightning:
        # Lightning mode with LoRA
        workflow["8"] = {
            "class_type": "KSampler",
            "inputs": {
                "model": ["12", 0],
                "positive": ["4", 0],
                "negative": ["9", 0],
                "latent_image": ["7", 0],
                "seed": seed,
                "steps": 4,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0
            }
        }
        workflow["12"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["5", 0],
                "lora_name": "Qwen-Image-Lightning-4steps-V1.0.safetensors",
                "strength_model": 1.0
            }
        }
    else:
        # Normal mode
        workflow["8"] = {
            "class_type": "KSampler",
            "inputs": {
                "model": ["5", 0],
                "positive": ["4", 0],
                "negative": ["9", 0],
                "latent_image": ["7", 0],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0
            }
        }

    return workflow

def queue_prompt(workflow):
    """Queue a prompt and return the prompt_id"""
    payload = {"prompt": workflow}
    try:
        r = requests.post(f"{COMFYUI_URL}/prompt", json=payload, timeout=10)
        if r.status_code == 200:
            return r.json().get("prompt_id")
    except Exception as e:
        print(f"Error queuing prompt: {e}")  # noqa: T201
    return None

def wait_for_completion(prompt_id, timeout=300):
    """Wait for prompt to complete, return elapsed time"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=5)
            if r.status_code == 200:
                history = r.json()
                if prompt_id in history:
                    return time.time() - start
        except:
            pass
        time.sleep(0.5)
    return None

def run_benchmark(resolutions, lightning=True, runs=3):
    """Run benchmarks and return results"""
    results = []
    mode = "Lightning" if lightning else "Normal"

    for res_name, width, height in resolutions:
        print(f"\n  Testing {res_name} ({mode})...")  # noqa: T201
        times = []

        for run in range(runs):
            workflow = build_workflow(width, height, TEST_PROMPT, lightning=lightning)
            prompt_id = queue_prompt(workflow)

            if prompt_id:
                elapsed = wait_for_completion(prompt_id)
                if elapsed:
                    times.append(elapsed)
                    print(f"    Run {run+1}: {elapsed:.2f}s")  # noqa: T201
                else:
                    print(f"    Run {run+1}: TIMEOUT")  # noqa: T201
            else:
                print(f"    Run {run+1}: FAILED to queue")  # noqa: T201

        if times:
            avg = sum(times) / len(times)
            results.append({
                "resolution": res_name,
                "mode": mode,
                "avg_time": avg,
                "min_time": min(times),
                "max_time": max(times),
                "runs": len(times)
            })

    return results

def print_results(results, title="Benchmark Results"):
    """Pretty print benchmark results"""
    print(f"\n{'='*60}")  # noqa: T201
    print(f" {title}")  # noqa: T201
    print(f"{'='*60}")  # noqa: T201
    print(f"{'Resolution':<12} {'Mode':<10} {'Avg':<8} {'Min':<8} {'Max':<8} {'Runs':<6}")  # noqa: T201
    print(f"{'-'*60}")  # noqa: T201

    for r in results:
        print(f"{r['resolution']:<12} {r['mode']:<10} {r['avg_time']:.2f}s    {r['min_time']:.2f}s    {r['max_time']:.2f}s    {r['runs']}")  # noqa: T201

    print(f"{'='*60}")  # noqa: T201

def save_results(results, filename=None):
    """Save results to JSON file"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_{timestamp}.json"

    with open(filename, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "prompt": TEST_PROMPT,
            "results": results
        }, f, indent=2)

    print(f"\nResults saved to: {filename}")  # noqa: T201

def main():
    parser = argparse.ArgumentParser(description="Qwen Image Generator Benchmark")
    parser.add_argument("--quick", action="store_true", help="Quick test (512x512 only)")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per test")
    parser.add_argument("--lightning-only", action="store_true", help="Only test lightning mode")
    parser.add_argument("--normal-only", action="store_true", help="Only test normal mode")
    parser.add_argument("--save", action="store_true", help="Save results to JSON")
    args = parser.parse_args()

    print("="*60)  # noqa: T201
    print(" Qwen Image Generator Benchmark")  # noqa: T201
    print("="*60)  # noqa: T201

    # Check ComfyUI
    print("\nChecking ComfyUI...")  # noqa: T201
    if not check_comfyui():
        print("ERROR: ComfyUI not running at", COMFYUI_URL)  # noqa: T201
        print("Start ComfyUI first, then run this benchmark.")  # noqa: T201
        return
    print("ComfyUI is running ✓")  # noqa: T201

    # Get available models
    models = get_current_model()
    print(f"Available models: {models}")  # noqa: T201

    # Select resolutions
    resolutions = QUICK_RESOLUTIONS if args.quick else RESOLUTIONS

    lightning_results = []
    normal_results = []

    # Run Lightning benchmarks (default: ON)
    if not args.normal_only:
        print("\n" + "-"*60)  # noqa: T201
        print(" ⚡ Lightning Mode ON (4 steps, CFG=1)")  # noqa: T201
        print("-"*60)  # noqa: T201
        lightning_results = run_benchmark(resolutions, lightning=True, runs=args.runs)

    # Run Normal benchmarks (default: ON)
    if not args.lightning_only:
        print("\n" + "-"*60)  # noqa: T201
        print(" 🎨 Lightning Mode OFF (30 steps, CFG=5)")  # noqa: T201
        print("-"*60)  # noqa: T201
        normal_results = run_benchmark(resolutions, lightning=False, runs=args.runs)

    # Print results
    all_results = lightning_results + normal_results
    print_results(all_results)

    # Print comparison if both modes tested
    if lightning_results and normal_results:
    print("\n" + "="*60)  # noqa: T201
    print(" ⚡ vs 🎨 Speed Comparison")  # noqa: T201
    print("="*60)  # noqa: T201
        for lr in lightning_results:
            for nr in normal_results:
                if lr['resolution'] == nr['resolution']:
                    speedup = nr['avg_time'] / lr['avg_time']
                    print(f"{lr['resolution']}: Lightning is {speedup:.1f}x faster")  # noqa: T201
                    print(f"  Lightning: {lr['avg_time']:.2f}s | Normal: {nr['avg_time']:.2f}s")  # noqa: T201

    # Save if requested
    if args.save:
        save_results(all_results)

    print("\nBenchmark complete!")  # noqa: T201

if __name__ == "__main__":
    main()
