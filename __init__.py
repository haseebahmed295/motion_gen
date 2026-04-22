bl_info = {
    "name": "Motion Gen",
    "author": "haseebahmed295",
    "version": (1, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > HY-Motion",
    "description": "Generate motion from prompt using HY-Motion and load into Blender.",
    "warning": "",
    "doc_url": "",
    "category": "Animation",
}

import bpy
import os
import sys
import subprocess
import threading
import time

def install_runtime_thread(operator_instance, python_exe, install_script):
    try:
        print("Starting HY-Motion Runtime Installation...")
        operator_instance._status = "Installing Python 3.11 Embedded Runtime... Check System Console."
        
        process = subprocess.Popen(
            [python_exe, install_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding='utf-8',
            errors='replace'
        )
        for line in process.stdout:
            print(line, end='', flush=True)
            
        process.wait()
        
        if process.returncode != 0:
            operator_instance._error = True
            operator_instance._status = "Installation Failed! Check console."
        else:
            operator_instance._status = "Installation Complete! Ready to Generate."
    except Exception as e:
        print("Error running installation:", str(e))
        operator_instance._error = True
        operator_instance._status = f"Error: {str(e)}"


def generate_motion_thread(operator_instance, python_exe, run_d_py, prompt, duration, seed, output_fbx, steps, cfg_scale, force_cpu):
    try:
        print("Starting HY-Motion generation process...")
        args_list = [
            python_exe, run_d_py, 
            "--prompt", prompt, 
            "--duration", str(duration), 
            "--seed", str(seed), 
            "--output_fbx", output_fbx, 
            "--steps", str(steps), 
            "--cfg", str(cfg_scale)
        ]
        
        if force_cpu:
            args_list.append("--force_cpu")
            
        process = subprocess.Popen(
            args_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding='utf-8',
            errors='replace'
        )
        for line in process.stdout:
            print(line, end='')
            s = line.strip()
            if "Loading Network" in s:
                operator_instance._log_msg = "Loading Diffusion Model..."
            elif "Starting Diffusion" in s:
                operator_instance._log_msg = "Generating Motion (Diffusion)..."
            elif "Encoding:" in s or "Encoding prompt:" in s:
                operator_instance._log_msg = "Encoding Prompt Context..."
            elif "Loading CLIP" in s or "Loading Qwen" in s:
                operator_instance._log_msg = "Loading Language Tokenizers..."
            elif "Generating Native Blender" in s:
                operator_instance._log_msg = "Exporting Motion Data..."
            elif "VRAM cleared" in s:
                operator_instance._log_msg = "Preparing GPU VRAM..."
            elif "Saved raw motion data" in s:
                operator_instance._log_msg = "Finalizing Motion Data..."
            elif "Using device:" in s:
                operator_instance._log_msg = "Initializing Compute Engine..."
        process.wait()
        
        if process.returncode != 0:
            operator_instance._error = True
    except Exception as e:
        print("Error running generation:", str(e))
        operator_instance._error = True

class HYMOTION_OT_generate(bpy.types.Operator):
    """Generate and Load Motion"""
    bl_idname = "hymotion.generate_and_load"
    bl_label = "Generate & Load Motion"
    bl_options = {'REGISTER'}

    _timer = None
    _thread = None
    _error = False
    _output_fbx = ""
    _start_time = 0.0
    _log_msg = ""

    def modal(self, context, event):
        if event.type == 'TIMER':
            if self._thread and self._thread.is_alive():
                context.scene.hy_motion_elapsed_time = time.time() - self._start_time
                context.scene.hy_motion_log_msg = self._log_msg
                for area in context.window.screen.areas:
                    if area.type == 'VIEW_3D':
                        area.tag_redraw()
                return {'PASS_THROUGH'}
            else:
                self.finish(context)
                if self._error:
                    self.report({'ERROR'}, "Motion Generation Failed. Check Console.")
                    return {'CANCELLED'}
                else:
                    self.apply_motion(context)
                    self.report({'INFO'}, "Motion Generated and Loaded Successfully.")
                    return {'FINISHED'}

        return {'PASS_THROUGH'}

    def execute(self, context):
        if context.scene.hy_motion_status == "Generating...":
            self.report({'WARNING'}, "Already generating...")
            return {'CANCELLED'}

        props = context.scene.hy_motion_props
        
        # Get addon dir
        import platform
        addon_dir = os.path.dirname(os.path.realpath(__file__))
        
        if platform.system() == "Windows":
            python_exe = os.path.join(addon_dir, "runtime", "python.exe")
        else:
            python_exe = os.path.join(addon_dir, "runtime", "bin", "python")
            
        run_d_py = os.path.join(addon_dir, "run_d.py")
        self._output_fbx = os.path.join(addon_dir, "motion_output.fbx")

        if not os.path.exists(python_exe):
            self.report({'ERROR'}, f"Python runtime not found at: {python_exe}")
            return {'CANCELLED'}
        
        if not os.path.exists(run_d_py):
            self.report({'ERROR'}, f"run_d.py script not found at: {run_d_py}")
            return {'CANCELLED'}

        context.scene.hy_motion_status = "Generating..."
        self._start_time = time.time()
        context.scene.hy_motion_elapsed_time = 0.0
        context.scene.hy_motion_log_msg = ""
        self._log_msg = ""
        self._error = False
        
        final_steps = 20 if props.draft_mode else props.steps

        prefs = context.preferences.addons[__name__].preferences
        force_cpu = prefs.force_cpu if prefs else False

        self._thread = threading.Thread(target=generate_motion_thread, args=(
            self, python_exe, run_d_py, props.prompt, props.duration, props.seed, self._output_fbx, final_steps, props.cfg_scale, force_cpu
        ))
        self._thread.start()

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.5, window=context.window)
        wm.modal_handler_add(self)

        return {'RUNNING_MODAL'}

    def finish(self, context):
        wm = context.window_manager
        if self._timer:
            wm.event_timer_remove(self._timer)
        context.scene.hy_motion_status = "Idle"
        for area in context.window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()

    def apply_motion(self, context):
        import json
        
        addon_dir = os.path.dirname(os.path.realpath(__file__))
        json_path = os.path.join(addon_dir, "motion_output.json")
        
        if not os.path.exists(json_path):
            self.report({'ERROR'}, f"JSON Payload not found at {json_path}")
            return
            
        with open(json_path, 'r') as f:
            motion_data = json.load(f)
            
        template_path = os.path.join(addon_dir, "assets", "wooden_models", "boy_Rigging_smplx_tex.fbx")
        if not os.path.exists(template_path):
            self.report({'ERROR'}, f"Character template FBX not found at {template_path}")
            return
            
        existing_objects = set(context.scene.objects)
        
        # 1. Import Template
        bpy.ops.import_scene.fbx(filepath=template_path)
        
        # 2. Locate the imported Armature
        new_objects = set(context.scene.objects) - existing_objects
        armature = None
        for obj in new_objects:
            if obj.type == 'ARMATURE':
                armature = obj
                break
                
        if not armature:
            self.report({'ERROR'}, "No Armature returned from template.")
            return

        context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode='POSE')
        
        frames = motion_data["length"]
        fps = motion_data["fps"]
        context.scene.render.fps = fps
        context.scene.frame_start = 1
        context.scene.frame_end = frames
        
        transl = motion_data["root_translation"]
        rotations = motion_data["joint_rotations"]
        
        for pb in armature.pose.bones:
            pb.rotation_mode = 'QUATERNION'
            
        def get_bone(name):
            if name in armature.pose.bones: return armature.pose.bones[name]
            for b in armature.pose.bones:
                if b.name.lower() == name.lower(): return b
            return None

        # 3. Apply Pelvis Translations
        pelvis_bone = get_bone("Pelvis")
        if pelvis_bone:
            for f_idx in range(frames):
                if f_idx < len(transl):
                    tx, ty, tz = transl[f_idx]
                    pelvis_bone.location = (tx, ty, tz)
                    pelvis_bone.keyframe_insert(data_path="location", frame=f_idx + 1)
                    
        # 4. Apply Bone Rotations
        for joint_name, quat_list in rotations.items():
            bone = get_bone(joint_name)
            if not bone: continue
            
            for f_idx in range(min(frames, len(quat_list))):
                q = quat_list[f_idx] # w, x, y, z
                bone.rotation_quaternion = (q[0], q[1], q[2], q[3])
                bone.keyframe_insert(data_path="rotation_quaternion", frame=f_idx + 1)
                
        bpy.ops.object.mode_set(mode='OBJECT')


class HYMOTION_OT_install_runtime(bpy.types.Operator):
    """Install standalone Python Environment for ML Models"""
    bl_idname = "hymotion.install_runtime"
    bl_label = "Install HY-Motion Backend"
    
    _timer = None
    _thread = None
    _error = False
    _status = "Idle"

    def modal(self, context, event):
        if event.type == 'TIMER':
            context.scene.hy_install_status = self._status
            
            # force update preferences panel
            for area in context.window.screen.areas:
                if area.type == 'PREFERENCES':
                    area.tag_redraw()

            if self._thread and self._thread.is_alive():
                return {'PASS_THROUGH'}
            else:
                wm = context.window_manager
                wm.event_timer_remove(self._timer)
                if self._error:
                    self.report({'ERROR'}, "Installation Failed. Check System Console.")
                else:
                    self.report({'INFO'}, "Successfully deployed HY-Motion ML runtime.")
                return {'FINISHED'}
                
        return {'PASS_THROUGH'}

    def execute(self, context):
        if context.scene.hy_install_status.startswith("Installing"):
            self.report({'WARNING'}, "Installation is already running...")
            return {'CANCELLED'}
            
        addon_dir = os.path.dirname(os.path.realpath(__file__))
        install_script = os.path.join(addon_dir, "install_env.py")
        blender_python = sys.executable
        
        if not os.path.exists(install_script):
            self.report({'ERROR'}, f"install_env.py not found at {install_script}")
            return {'CANCELLED'}
            
        self._error = False
        self._status = "Installing... Please wait and check console."
        context.scene.hy_install_status = self._status
        
        self._thread = threading.Thread(target=install_runtime_thread, args=(
            self, blender_python, install_script
        ))
        self._thread.start()
        
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.5, window=context.window)
        wm.modal_handler_add(self)
        
        return {'RUNNING_MODAL'}




def copy_models_thread(operator_instance, addon_dir, ckpt_src, gguf_src):
    try:
        import shutil
        operator_instance._status = "Copying Models..."
        
        if ckpt_src and os.path.isfile(ckpt_src):
            operator_instance._status = "Copying latest.ckpt..."
            dest_dir = os.path.join(addon_dir, "HY-Motion-1.0")
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(ckpt_src, os.path.join(dest_dir, "latest.ckpt"))
            
        if gguf_src and os.path.isfile(gguf_src):
            operator_instance._status = "Copying Qwen GGUF..."
            dest_dir = os.path.join(addon_dir, "GGUF")
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(gguf_src, os.path.join(dest_dir, "Qwen3-8B-UD-Q5_K_XL.gguf"))
            
        operator_instance._status = "Complete!"
    except Exception as e:
        print("Copy Error:", e)
        operator_instance._error = True
        operator_instance._status = f"Error: {e}"

class HYMOTION_OT_import_models(bpy.types.Operator):
    """Copy selected models into the Addon folder"""
    bl_idname = "hymotion.import_models"
    bl_label = "Import Selected Models to Addon"
    
    _timer = None
    _thread = None
    _error = False
    _status = "Idle"

    def modal(self, context, event):
        if event.type == 'TIMER':
            context.scene.hy_install_status = self._status
            
            for area in context.window.screen.areas:
                if area.type == 'PREFERENCES':
                    area.tag_redraw()

            if self._thread and self._thread.is_alive():
                return {'PASS_THROUGH'}
            else:
                wm = context.window_manager
                wm.event_timer_remove(self._timer)
                if self._error:
                    self.report({'ERROR'}, "Import Failed. Check System Console.")
                else:
                    self.report({'INFO'}, "Successfully imported AI models.")
                    prefs = context.preferences.addons[__name__].preferences
                    prefs.source_ckpt_path = ""
                    prefs.source_gguf_path = ""
                return {'FINISHED'}
                
        return {'PASS_THROUGH'}

    def execute(self, context):
        prefs = context.preferences.addons[__name__].preferences
        addon_dir = os.path.dirname(os.path.realpath(__file__))
        
        ckpt_src = bpy.path.abspath(prefs.source_ckpt_path)
        gguf_src = bpy.path.abspath(prefs.source_gguf_path)
        
        if not ckpt_src and not gguf_src:
            self.report({'WARNING'}, "Please select at least one file to import.")
            return {'CANCELLED'}
            
        self._error = False
        self._status = "Starting Import..."
        context.scene.hy_install_status = self._status
        
        self._thread = threading.Thread(target=copy_models_thread, args=(
            self, addon_dir, ckpt_src, gguf_src
        ))
        self._thread.start()
        
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.5, window=context.window)
        wm.modal_handler_add(self)
        
        return {'RUNNING_MODAL'}


class HYMotionPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    force_cpu: bpy.props.BoolProperty(
        name="Force CPU Execution",
        description="Run the entire pipeline on CPU instead of GPU. Extremely slow, only use for debugging or on incompatible hardware.",
        default=False
    )
    
    source_ckpt_path: bpy.props.StringProperty(
        name="Select latest.ckpt",
        description="Path to the downloaded latest.ckpt file",
        subtype='FILE_PATH',
        default=""
    )

    source_gguf_path: bpy.props.StringProperty(
        name="Select Qwen GGUF",
        description="Path to the downloaded .gguf file",
        subtype='FILE_PATH',
        default=""
    )

    def draw(self, context):
        layout = self.layout
        box = layout.box()
        box.label(text="Local Setup", icon='PREFERENCES')
        box.prop(self, "force_cpu")
        box.label(text="HY-Motion uses a disconnected Python 3.11 instance to prevent dependency conflicts with Blender.")
        
        import platform
        import glob
        addon_dir = os.path.dirname(os.path.realpath(__file__))
        runtime_dir = os.path.join(addon_dir, "runtime")
        
        if platform.system() == "Windows":
            python_exe = os.path.join(runtime_dir, "python.exe")
            torch_path = os.path.join(runtime_dir, "Lib", "site-packages", "torch")
            torch_installed = os.path.exists(torch_path)
        else:
            python_exe = os.path.join(runtime_dir, "bin", "python")
            torch_paths = glob.glob(os.path.join(runtime_dir, "lib", "python*", "site-packages", "torch"))
            torch_installed = len(torch_paths) > 0
        
        status = context.scene.hy_install_status
        is_installing = status.startswith("Installing")
        
        if not is_installing:
            if not os.path.exists(python_exe):
                env_state = "RUNTIME_MISSING"
            elif not torch_installed:
                env_state = "MODULES_MISSING"
            else:
                env_state = "READY"
        else:
            env_state = "INSTALLING"

        row = box.row()
        
        if env_state == "INSTALLING":
            row.enabled = False
            row.operator("hymotion.install_runtime", text=status, icon='TIME')
            box.label(text=status, icon='INFO')
        elif env_state == "RUNTIME_MISSING":
            row.operator("hymotion.install_runtime", text="Install Python Runtime", icon='CONSOLE')
            box.label(text="Python 3.11 Runtime is missing.", icon='ERROR')
        elif env_state == "MODULES_MISSING":
            row.operator("hymotion.install_runtime", text="Install Missing Modules", icon='CONSOLE')
            box.label(text="Runtime found, but ML modules are missing.", icon='ERROR')
        else: # READY
            row.operator("hymotion.install_runtime", text="Reinstall / Repair Environment", icon='FILE_TICK')
            box.label(text="Runtime and Dependencies Installed!", icon='CHECKMARK')

        # Model download box
        layout.separator()
        model_box = layout.box()
        model_box.label(text="AI Model Paths (Manual Download Required)", icon='NODE_MATERIAL')
        
        # Check files
        hymotion_ckpt = os.path.join(addon_dir, "HY-Motion-1.0", "latest.ckpt")
        qwen_gguf = os.path.join(addon_dir, "GGUF", "Qwen3-8B-UD-Q5_K_XL.gguf")
        
        models_ready = os.path.exists(hymotion_ckpt) and os.path.exists(qwen_gguf)
        
        if not models_ready:
            model_box.label(text="AI Model binaries are missing.", icon='ERROR')
            model_box.label(text="Please select the downloaded files below to import them into the Addon:", icon='INFO')
            
            # Add string property locators to UI
            model_box.prop(self, "source_ckpt_path")
            model_box.prop(self, "source_gguf_path")
            
            if status.startswith("Copying") or status.startswith("Starting Import"):
                model_box.label(text=status, icon='TIME')
            elif self.source_ckpt_path or self.source_gguf_path:
                model_box.operator("hymotion.import_models", text="Import Selected Models", icon='IMPORT')
            
            model_box.separator()
            model_box.label(text="Need to download them?", icon='QUESTION')
            links_row = model_box.row()
            links_row.operator("wm.url_open", text="Download latest.ckpt (1.8GB)", icon='URL').url = "https://huggingface.co/tencent/HY-Motion-1.0/resolve/main/HY-Motion-1.0/latest.ckpt"
            links_row.operator("wm.url_open", text="Download Qwen GGUF (5.8GB)", icon='URL').url = "https://huggingface.co/unsloth/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-UD-Q5_K_XL.gguf"
        else:
            model_box.label(text="All required models found locally!", icon='CHECKMARK')


class HYMotionProperties(bpy.types.PropertyGroup):
    prompt: bpy.props.StringProperty(
        name="Prompt",
        description="Describe the motion exactly as you want it generated (e.g., 'A person doing a backflip')",
        default="A person jumping over a low wall"
    )
    duration: bpy.props.FloatProperty(
        name="Duration (s)",
        description="Total length of the animation. Warning: Generation time scales linearly with duration.",
        default=3.0,
        min=0.5,
        max=30.0
    )
    draft_mode: bpy.props.BoolProperty(
        name="Draft Mode",
        description="Fast prototyping mode. Locks Inference Steps to 20 for rapid previews.",
        default=False
    )
    steps: bpy.props.IntProperty(
        name="Steps",
        description="Quality vs Speed. Draft (20-30), Production (50-60). Higher means smoother physics but slower render.",
        default=50,
        min=10,
        max=100
    )
    cfg_scale: bpy.props.FloatProperty(
        name="CFG Scale",
        description="Adherence vs Flow. Low (2-4) favors physics, High (5-7) forces prompt exactness. >8 may cause jitters.",
        default=5.0,
        min=1.0,
        max=10.0
    )
    seed: bpy.props.IntProperty(
        name="Seed",
        description="Iteration dial. Leave at -1 to get a random variation every time, or type a number to lock it in.",
        default=-1,
        min=-1
    )

class HYMOTION_PT_main_panel(bpy.types.Panel):
    """Creates a Panel in the scene context of the properties editor"""
    bl_label = "HY-Motion Generator"
    bl_idname = "HYMOTION_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'HY-Motion'

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        props = scene.hy_motion_props

        # --- Prompt Section ---
        box = layout.box()
        box.label(text="Motion Prompt", icon='TEXT')
        box.prop(props, "prompt", text="")
        
        row = box.row()
        row.prop(props, "duration", slider=True)
        
        layout.separator()
        
        # --- Advanced Settings Section ---
        box = layout.box()
        box.label(text="Generation Settings", icon='PREFERENCES')
        col = box.column(align=True)
        col.prop(props, "draft_mode", icon='SHADING_BBOX')
        
        row = col.row()
        row.prop(props, "steps", slider=True)
        row.enabled = not props.draft_mode
        
        col.prop(props, "cfg_scale", slider=True)
        col.prop(props, "seed")

        layout.separator()

        # --- Status & Actions ---
        addon_dir = os.path.dirname(os.path.realpath(__file__))
        site_packages = os.path.join(addon_dir, "runtime", "Lib", "site-packages")
        python_exe = os.path.join(addon_dir, "runtime", "python.exe")
        env_ready = os.path.exists(python_exe) and os.path.exists(os.path.join(site_packages, "torch"))

        if not env_ready:
            layout.alert = True
            layout.operator("hymotion.generate_and_load", text="Setup Environment in Preferences", icon='ERROR')
        elif scene.hy_motion_status == "Generating...":
            t = scene.hy_motion_elapsed_time
            formatted_time = f"{int(t // 60):02d}:{int(t % 60):02d}"
            
            box = layout.box()
            box.label(text=f"Generating... {formatted_time}", icon='TIME')
            if scene.hy_motion_log_msg:
                row = box.row()
                row.label(text=scene.hy_motion_log_msg, icon='CONSOLE')
                
            row = box.row()
            row.enabled = False
            row.scale_y = 1.5
            row.operator("hymotion.generate_and_load", text="Processing (Wait)", icon='GHOST_ENABLED')
        else:
            row = layout.row()
            row.scale_y = 1.5
            row.operator("hymotion.generate_and_load", text="Generate & Load Motion", icon='PLAY')

            if scene.hy_motion_elapsed_time > 0 and scene.hy_motion_status == "Idle":
                t = scene.hy_motion_elapsed_time
                formatted_time = f"{int(t // 60):02d}:{int(t % 60):02d}"
                layout.separator()
                box = layout.box()
                box.label(text=f"Last generation took: {formatted_time}", icon='INFO')

classes = (
    HYMotionProperties,
    HYMOTION_OT_generate,
    HYMOTION_PT_main_panel,
    HYMOTION_OT_install_runtime,
    HYMOTION_OT_import_models,
    HYMotionPreferences,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.hy_motion_props = bpy.props.PointerProperty(type=HYMotionProperties)
    bpy.types.Scene.hy_motion_status = bpy.props.StringProperty(default="Idle")
    bpy.types.Scene.hy_install_status = bpy.props.StringProperty(default="Idle")
    bpy.types.Scene.hy_motion_elapsed_time = bpy.props.FloatProperty(default=0.0)
    bpy.types.Scene.hy_motion_log_msg = bpy.props.StringProperty(default="")

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.hy_motion_props
    del bpy.types.Scene.hy_motion_status
    del bpy.types.Scene.hy_install_status
    del bpy.types.Scene.hy_motion_elapsed_time
    del bpy.types.Scene.hy_motion_log_msg

if __name__ == "__main__":
    register()
