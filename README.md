# HY-Motion Blender Addon

Generate 3D motion from text prompts directly within Blender using the HY-Motion diffusion model and apply it to a built-in character template. 

## Features
- **Prompt-to-Motion:** Describe an animation (e.g., "A person doing a backflip") and generate it in seconds.
- **Isolated ML Runtime:** Uses a standalone Python environment to ensure it doesn't conflict with Blender's internal dependencies.
- **Auto-Rigging:** Automatically applies the generated motion data to a character armature.

## Installation Guide

### Step 1: Install the Addon in Blender
1. Download the `motion_gen.zip` file (do not extract it).
2. Open Blender (version 4.2.0 or higher recommended).
3. Go to **Edit > Preferences > Add-ons**.
4. Click on the **Install...** button at the top right.
5. Locate and select the `motion_gen.zip` file, then click **Install Add-on**.
6. Enable the addon by checking the box next to **Animation: Motion Gen**.

### Step 2: Install the HY-Motion Backend
1. Once enabled, expand the addon details by clicking the arrow next to its name.
2. In the **Local Setup** section, you will see the status of the Python runtime environment.
3. Click the **Install Python Runtime** (or **Install HY-Motion Backend**) button. This will configure a standalone Python instance just for the addon.
4. Wait for the installation to finish. You can check the Blender system console (`Window > Toggle System Console`) for progress.
5. Wait until the status says **Runtime and Dependencies Installed!**.

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
