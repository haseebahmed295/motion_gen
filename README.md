# HY-Motion Blender Addon

Generate 3D motion from text prompts directly within Blender using the HY-Motion diffusion model and apply it to a built-in character template. 

## Features
- **Prompt-to-Motion:** Describe an animation (e.g., "A person doing a backflip") and generate it in seconds.
- **Isolated ML Runtime:** Uses a standalone Python environment to ensure it doesn't conflict with Blender's internal dependencies.
- **Auto-Rigging:** Automatically applies the generated motion data to a character armature.
- **Hardware Acceleration:** Supports NVIDIA CUDA on Windows and native Metal Performance Shaders (MPS) / Metal GPU on Apple Silicon Macs.

## System Requirements

| Platform | Recommended Specs | Minimum Specs |
| :--- | :--- | :--- |
| **macOS (Apple Silicon)** | M1/M2/M3/M4, **24 GB+ Unified Memory**, macOS 12+ | M1/M2, **16 GB Unified Memory** (close heavy apps) |
| **Windows 10 / 11** | NVIDIA RTX 3060+ (8 GB+ VRAM), 32 GB RAM | NVIDIA GTX/RTX (8 GB VRAM), 16 GB RAM |

## Installation Guide

### Step 1: Install the Addon in Blender
1. Download the addon `.zip` file for your platform:
   - **For macOS (Apple Silicon M1/M2/M3/M4):** `motion_gen-macos-arm64.zip` (or universal `motion_gen.zip`)
   - **For Windows:** `motion_gen.zip`
2. Open Blender (version 4.2.0 or higher recommended).
3. Go to **Edit > Preferences > Add-ons** (or **Get Extensions > Install from Disk...** in Blender 4.2+).
4. Locate and select the downloaded `.zip` file, then click **Install Add-on**.
5. Enable the addon by checking the box next to **Animation: Motion Gen**.

### Step 2: Install the HY-Motion Backend
1. Once enabled, expand the addon details by clicking the arrow next to its name.
2. In the **Local Setup** section, you will see the status of the Python runtime environment.
3. Click the **Install Python Runtime** (or **Install HY-Motion Backend**) button:
   - **On macOS (Apple Silicon):** Automatically deploys a standalone Python 3.11 environment with native PyTorch (MPS GPU) and the bundled Metal-accelerated `llama-cpp-python` wheel.
   - **On Windows:** Automatically deploys Python 3.11 with CUDA GPU acceleration.
4. Wait for the installation to finish. You can check the Blender system console (`Window > Toggle System Console` on Windows, or launch Blender from Terminal on macOS) for progress.
5. Wait until the status says **Runtime and Dependencies Installed!**.

> [!TIP]
> **macOS Build Tools Fallback:** A pre-compiled Apple Silicon Metal wheel is bundled with the Mac release. If you ever need to rebuild dependencies from source on macOS, ensure you have Xcode Command Line Tools (`xcode-select --install`) and CMake (`brew install cmake`) installed.

### Step 3: Download & Import AI Models
Due to their large size, the AI models must be downloaded separately.

1. In the addon preferences under **AI Model Paths (Manual Download Required)**, click the provided links to download the models:
   - **Download latest.ckpt (1.8GB)**
   - **Download Qwen GGUF (5.8GB)**
2. Once the files are downloaded to your computer, use the file selectors in the addon preferences:
   - For **Select latest.ckpt**, browse to and choose the downloaded `latest.ckpt` file.
   - For **Select Qwen GGUF**, browse to and choose the downloaded `Qwen3-8B-UD-Q5_K_XL.gguf` file.
3. Click **Import Selected Models**. The addon will copy these files to the correct internal directories.
4. When finished, it will display **All required models found locally!**

## Usage

1. Open the 3D Viewport in Blender.
2. Press `N` to open the Sidebar and go to the **HY-Motion** tab.
3. **Motion Prompt:** Enter a description for your animation.
4. **Duration:** Adjust the length of the animation in seconds.
5. **Generation Settings:** Tweak the draft mode, inference steps, and CFG scale as needed.
6. Click **Generate & Load Motion**.
7. Wait for the generation to finish. The addon will automatically import a character model and apply the new motion directly to its armature.
