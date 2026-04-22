import os
import sys
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from hymotion.network.text_encoders.model_constants import PROMPT_TEMPLATE_ENCODE_HUMAN_MOTION
from hymotion.utils.loaders import load_object
from hymotion.pipeline.motion_diffusion import length_to_mask, randn_tensor
import torch
import numpy as np
from torchdiffeq import odeint

class LlamaCppModelWrapper:
    def __init__(self, model_path, n_gpu_layers=-1):
        try:
            from llama_cpp import Llama
            self.llm = Llama(
                model_path=model_path,
                embedding=True,
                n_ctx=2048,
                n_batch=2048,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
                logits_all=False
            )
            self.config = type('obj', (object,), {'hidden_size': 4096})
            self.dummy_param = torch.nn.Parameter(torch.empty(0))
        except ImportError:
            raise ImportError(
                "Please install llama-cpp-python in your runtime.")

    def parameters(self):
        return iter([self.dummy_param])

    def __call__(self, input_ids, attention_mask=None, output_hidden_states=True):
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        all_batch_embeddings = []

        for i in range(batch_size):
            tokens = input_ids[i].tolist()
            self.llm.reset()
            print(
                f"[HY-Motion] Encoding prompt: Batch processing {len(tokens)} tokens...")
            self.llm.eval(tokens)

            seq_embeddings = []
            for idx in range(seq_len):
                emb_ptr = self.llm._ctx.get_embeddings_ith(idx)
                token_vec = [float(emb_ptr[j]) for j in range(4096)]
                seq_embeddings.append(token_vec)

            all_batch_embeddings.append(seq_embeddings)

        embeddings_tensor = torch.tensor(
            all_batch_embeddings, dtype=torch.float16)
        return type('obj', (object,), {'hidden_states': [embeddings_tensor]})

    def eval(self): return self
    def requires_grad_(self, val): return self

    def to(self, device):
        self.dummy_param.data = self.dummy_param.data.to(device)
        return self


def generate_pure_python(prompt, duration=3.0, seed=42, output_fbx="motion_output.fbx", steps=50, cfg_scale=5.0, force_cpu=False):
    if seed == -1:
        import random
        seed = random.randint(1, 100000)
    device = torch.device("cpu") if force_cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------------------------------------------------------
    # 1. LOAD TOKENIZERS & TEXT ENCODERS
    # ---------------------------------------------------------
    from transformers import CLIPTokenizer, CLIPTextModel, AutoTokenizer

    base_dir = os.path.dirname(os.path.realpath(__file__))
    clip_path = os.path.join(base_dir, "clip-vit-large-patch14")
    qwen_tok_path = os.path.join(base_dir, "Qwen3-8B")
    gguf_path = os.path.join(base_dir, "GGUF", "Qwen3-8B-UD-Q5_K_XL.gguf")

    print("Loading CLIP...")
    clip_tokenizer = CLIPTokenizer.from_pretrained(
        clip_path, local_files_only=True)
    clip_model = CLIPTextModel.from_pretrained(
        clip_path, local_files_only=True).to(device).eval()

    print("Loading Qwen Tokenizer & Llama-CPP...")
    qwen_tokenizer = AutoTokenizer.from_pretrained(
        qwen_tok_path, padding_side="right", local_files_only=True)
    llm = LlamaCppModelWrapper(model_path=gguf_path, n_gpu_layers=0)

    # ---------------------------------------------------------
    # 2. ENCODE THE TEXT (CONDITIONING)
    # ---------------------------------------------------------
    print(f"Encoding: '{prompt}'")
    with torch.no_grad():
        # --- CLIP Encoding ---
        clip_enc = clip_tokenizer(
            [prompt], truncation=True, max_length=77, padding=True, return_tensors="pt")
        clip_out = clip_model(input_ids=clip_enc["input_ids"].to(
            device), attention_mask=clip_enc["attention_mask"].to(device))
        vtxt_raw = clip_out.pooler_output.unsqueeze(
            1) if clip_out.pooler_output is not None else clip_out.last_hidden_state.mean(1, keepdim=True)

        # Free CLIP VRAM
        del clip_model, clip_enc, clip_out
        torch.cuda.empty_cache()

        # --- Qwen/Llama-CPP Encoding ---
        # Calculate crop start to remove system prompt padding later
        marker = "<BOC>"
        msgs = [{"role": "system", "content": PROMPT_TEMPLATE_ENCODE_HUMAN_MOTION}, {
            "role": "user", "content": marker}]
        s = qwen_tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False)
        full_ids = qwen_tokenizer(s, return_tensors="pt")[
            "input_ids"][0].tolist()
        marker_ids = qwen_tokenizer(marker, return_tensors="pt", add_special_tokens=False)[
            "input_ids"][0].tolist()

        def find_subseq(a, b):
            for i in range(len(a) - len(b) + 1):
                if a[i:i + len(b)] == b:
                    return i
            return max(0, len(a) - 1)
        crop_start = find_subseq(full_ids, marker_ids)

        # Encode full prompt
        max_length = 512
        total_len = max_length + crop_start
        llm_text = qwen_tokenizer.apply_chat_template(
            [{"role": "system", "content": PROMPT_TEMPLATE_ENCODE_HUMAN_MOTION}, {
                "role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=False
        )
        llm_enc = qwen_tokenizer(
            [llm_text], truncation=True, max_length=total_len, padding="max_length", return_tensors="pt")

        llm_out = llm(input_ids=llm_enc["input_ids"])
        ctxt_raw = llm_out.hidden_states[-1][:,
                                             crop_start:crop_start + max_length].contiguous().to(device)
        ctxt_length = (llm_enc["attention_mask"].sum(
            dim=-1) - crop_start).clamp(0, max_length).to(device)

        print(
            f"Encoded Vectors -> vtxt: {vtxt_raw.shape}, ctxt: {ctxt_raw.shape}")
        
        print("Unloading Qwen from GPU and wiping VRAM...")
    
    import gc
    del llm.llm  # Kill the C++ backend
    del llm      # Kill the Python wrapper
    gc.collect()  # Force garbage collection
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()
    print("VRAM cleared. Safe to load Diffusion Model.")

    # ---------------------------------------------------------
    # 3. LOAD THE DIFFUSION NETWORK
    # ---------------------------------------------------------
    print("Loading Network...")
    net_args = {
        "apply_rope_to_single_branch": False, "ctxt_input_dim": 4096,
        "dropout": 0.0, "feat_dim": 1280, "input_dim": 201,
        "mask_mode": "narrowband", "mlp_ratio": 4.0,
        "num_heads": 20, "num_layers": 27,
        "time_factor": 1000.0, "vtxt_input_dim": 768,
    }
    network = load_object(
        "hymotion/network/hymotion_mmdit.HunyuanMotionMMDiT", net_args)

    model_dir = os.path.join(base_dir, "HY-Motion-1.0")
    ckpt_path = os.path.join(model_dir, "latest.ckpt")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    state_dict = {k.replace("motion_transformer.", ""): v for k,
                  v in ckpt["model_state_dict"].items() if k.startswith("motion_transformer.")}
    network.load_state_dict(state_dict, strict=False)
    network.to(device).eval()

    # Load baseline stats for generation/decoding
    net_mean = ckpt["model_state_dict"].get(
        "mean", torch.zeros(201)).to(device)
    net_std = ckpt["model_state_dict"].get("std", torch.ones(201)).to(device)
    null_vtxt_feat = ckpt["model_state_dict"].get(
        "null_vtxt_feat", torch.randn(1, 1, 768)).to(device)
    null_ctxt_input = ckpt["model_state_dict"].get(
        "null_ctxt_input", torch.randn(1, 1, 4096)).to(device)

    # ---------------------------------------------------------
    # 4. DIFFUSION LOOP (THE MATH)
    # ---------------------------------------------------------
    print("Starting Diffusion...")
    fps = 30
    length = min(max(int(duration * fps), 20), 360)  # Max 360 frames (12s)
    val_steps = steps

    vtxt_input = vtxt_raw
    ctxt_input = ctxt_raw

    ctxt_mask = length_to_mask(ctxt_length, ctxt_input.shape[1])
    x_length = torch.LongTensor([length]).to(device)
    x_mask = length_to_mask(x_length, 360)

    # Apply CFG (Classifier Free Guidance) structure
    vtxt_input = torch.cat(
        [null_vtxt_feat.expand_as(vtxt_input), vtxt_input], dim=0)
    ctxt_input = torch.cat(
        [null_ctxt_input.expand_as(ctxt_input), ctxt_input], dim=0)
    ctxt_mask = torch.cat([ctxt_mask] * 2, dim=0)
    x_mask = torch.cat([x_mask] * 2, dim=0)

    def fn(t, x):
        x_in = torch.cat([x] * 2, dim=0)
        pred = network(x=x_in, ctxt_input=ctxt_input, vtxt_input=vtxt_input,
                       timesteps=t.expand(x_in.shape[0]), x_mask_temporal=x_mask, ctxt_mask_temporal=ctxt_mask)
        pred_u, pred_c = pred.chunk(2)
        return pred_u + cfg_scale * (pred_c - pred_u)

    t = torch.linspace(0, 1, val_steps + 1, device=device)
    y0 = randn_tensor((1, 360, 201), generator=torch.Generator(
        device=device).manual_seed(seed), device=device)

    with torch.no_grad():
        # Solve the ODE
        sampled = odeint(fn, y0, t, method="euler")[-1][:, :length]

    # Apply standard deviation and mean to get actual physical latent limits
    std = torch.where(net_std < 1e-3, torch.ones_like(net_std), net_std)
    x_out = sampled * std + net_mean

    # Extract translation (first 3) and Rot6D coordinates
    transl = x_out[..., :3].cpu().numpy()
    rot6d = torch.cat([x_out[..., 3:9].reshape(1, length, 1, 6),
                      x_out[..., 9:9+126].reshape(1, length, 21, 6)], dim=2).cpu().numpy()

    print(f"Motion Generated successfully. (Shape: {rot6d.shape})")

    # Save the raw numpy arrays so your Blender script can read them!
    out_file = os.path.splitext(output_fbx)[0] + ".npz"
    np.savez(out_file, rot6d=rot6d, transl=transl,
             text=prompt, duration=duration)
    print(f"Saved raw motion data to {out_file}")

    print("Generating Native Blender JSON Payload...")
    try:
        import json
        from scipy.spatial.transform import Rotation as R
        from hymotion.utils.geometry import rot6d_to_rotation_matrix

        rot6d_tensor = torch.from_numpy(rot6d[0]).float() # (length, 22, 6)
        transl_tensor = torch.from_numpy(transl[0]).float()

        rot_matrices = rot6d_to_rotation_matrix(rot6d_tensor).numpy() # (length, 22, 3, 3)
        lengths = rot_matrices.shape[0]

        SMPLH_JOINTS = [
            "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2",
            "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck", "L_Collar",
            "R_Collar", "Head", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
            "L_Wrist", "R_Wrist"
        ]

        motion_data = {
            "fps": 30,
            "length": lengths,
            "root_translation": (transl_tensor * 100.0).tolist(), # Convert to cm scale
            "joint_rotations": {}
        }

        for i, joint_name in enumerate(SMPLH_JOINTS):
            mat = rot_matrices[:, i, :, :]
            r = R.from_matrix(mat)
            quats = r.as_quat() # x, y, z, w
            blender_quats = np.stack([quats[:, 3], quats[:, 0], quats[:, 1], quats[:, 2]], axis=1) # w, x, y, z
            motion_data["joint_rotations"][joint_name] = blender_quats.tolist()

        json_out = os.path.splitext(output_fbx)[0] + ".json"
        with open(json_out, "w") as f:
            json.dump(motion_data, f)
            
        print(f"Successfully saved JSON payload to {json_out}")

    except Exception as e:
        print(f"Error during JSON payload generation: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate human motion")
    parser.add_argument("--prompt", type=str, default="A person jumping over a low wall", help="Text prompt for the motion")
    parser.add_argument("--duration", type=float, default=3.0, help="Duration in seconds")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed (-1 for random)")
    parser.add_argument("--output_fbx", type=str, default="motion_output.fbx", help="Output path for the fbx")
    parser.add_argument("--steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--cfg", type=float, default=5.0, help="CFG scale")
    parser.add_argument("--force_cpu", action="store_true", help="Force the entire pipeline to run on CPU")
    args = parser.parse_args()
    
    generate_pure_python(args.prompt, duration=args.duration, seed=args.seed, output_fbx=args.output_fbx, steps=args.steps, cfg_scale=args.cfg, force_cpu=args.force_cpu)
