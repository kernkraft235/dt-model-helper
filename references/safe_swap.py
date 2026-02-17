#!/usr/bin/env python3
import sqlite3
import torch
import sys
import shutil
import os
from safetensors.torch import load_file

# --- CONFIGURATION ---
# Use the Big 8GB file as the donor
INPUT_DONOR_CKPT  = "qwen_3_vl_4b_instruct_f16.ckpt"
# Use your converted FP16 safetensors
INPUT_SAFETENSORS = "model.fp16.safetensors"
OUTPUT_NAME       = "qwen_3_vl_4b_heretic"

def map_to_dt(sf_key):
    # 1. Embeddings & Final Norm
    if sf_key == "model.embed_tokens.weight": return "__text_model__[t-tok_embeddings-0-0]"
    if sf_key == "model.norm.weight": return "__text_model__[t-norm-0-0]"

    # 2. Deep Layers
    if "model.layers." in sf_key:
        try:
            parts = sf_key.split(".")
            layer_idx = parts[2]
            suffix = ".".join(parts[3:])
            
            if "self_attn.q_proj" in suffix: return f"__text_model__[t-q_proj-{layer_idx}-0]"
            if "self_attn.k_proj" in suffix: return f"__text_model__[t-k_proj-{layer_idx}-0]"
            if "self_attn.v_proj" in suffix: return f"__text_model__[t-v_proj-{layer_idx}-0]"
            if "self_attn.o_proj" in suffix: return f"__text_model__[t-out_proj-{layer_idx}-0]"
            
            if "input_layernorm" in suffix:  return f"__text_model__[t-input_layernorm-{layer_idx}-0]"
            if "post_attention_layernorm" in suffix: return f"__text_model__[t-post_attention_layernorm-{layer_idx}-0]"
            
            if "mlp.gate_proj" in suffix: return f"__text_model__[t-mlp-{layer_idx}-mlp_gate_proj-0-0]"
            if "mlp.up_proj" in suffix:   return f"__text_model__[t-mlp-{layer_idx}-mlp_up_proj-0-0]"
            if "mlp.down_proj" in suffix: return f"__text_model__[t-mlp-{layer_idx}-mlp_down_proj-0-0]"
        except: pass
    return None

def main():
    print(f"[:] STARTING. Target: {OUTPUT_NAME}.ckpt", flush=True)
    
    # 1. Verification
    if not os.path.exists(INPUT_DONOR_CKPT):
        print(f"[!] Error: Donor file {INPUT_DONOR_CKPT} not found.", flush=True)
        sys.exit(1)
    if not os.path.exists(INPUT_SAFETENSORS):
        print(f"[!] Error: Weights file {INPUT_SAFETENSORS} not found.", flush=True)
        sys.exit(1)

    # 2. Load Weights
    print(f"[:] Loading {INPUT_SAFETENSORS}...", flush=True)
    sf_tensors = load_file(INPUT_SAFETENSORS)
    
    sf_map = {}
    for k, v in sf_tensors.items():
        dt_key = map_to_dt(k)
        if dt_key: sf_map[dt_key] = v
            
    print(f"[:] Mapped {len(sf_map)} tensors ready for injection.", flush=True)

    # 3. Clone Donor
    out_ckpt = f"{OUTPUT_NAME}.ckpt"
    print(f"[:] Cloning {INPUT_DONOR_CKPT} (This may take a minute)...", flush=True)
    if os.path.exists(out_ckpt): os.remove(out_ckpt)
    shutil.copy(INPUT_DONOR_CKPT, out_ckpt)
    
    # 4. Open DB
    conn = sqlite3.connect(out_ckpt)
    cursor = conn.cursor()
    
    # Verify Schema matches what you saw in 'head'
    cursor.execute("PRAGMA table_info(tensors)")
    cols = [r[1] for r in cursor.fetchall()]
    if 'data' not in cols:
        print("[!] FATAL: This script expects the 'data' BLOB column.", flush=True)
        sys.exit(1)
        
    print(f"[:] Schema Verified. Writing data to 'tensors' table...", flush=True)

    # 5. Inject Data
    cursor.execute("SELECT name FROM tensors")
    db_keys = [r[0] for r in cursor.fetchall()]
    
    updated_count = 0
    for key in db_keys:
        if key in sf_map:
            tensor = sf_map[key]
            
            # A. Clamp (Fixes Deep Fried / BF16 artifacts)
            clamped = torch.clamp(tensor.float(), min=-65500.0, max=65500.0)
            
            # B. Cast to F16
            data_bytes = clamped.to(torch.float16).numpy().tobytes()
            
            # C. Update BLOB
            # Note: We also set type=1 (F16) just to be safe, though donor is likely already 1.
            cursor.execute("UPDATE tensors SET data=?, type=1 WHERE name=?", (data_bytes, key))
            
            updated_count += 1
            if updated_count % 50 == 0: print(f"    Injected {updated_count} layers...", end="\r", flush=True)

    conn.commit()
    conn.close()
    
    print(f"\n[:] DONE. Updated {updated_count} layers.", flush=True)
    print(f"[:] Output: {out_ckpt}", flush=True)

if __name__ == "__main__":
    main()