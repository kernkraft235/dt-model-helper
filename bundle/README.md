# dt-model-helper

CLI tools for importing LoRA models into [Draw Things](https://drawthings.ai) without the GUI. Handles safetensors header fixing, CivitAI metadata lookup, format conversion, and model registration — all in one command.

## What it does

When you download a LoRA `.safetensors` file from CivitAI (or anywhere else), Draw Things can't use it directly. It needs to be converted to a `.ckpt` (SQLite) format and registered in `custom_lora.json`. This tool automates the full pipeline:

1. Computes SHA256 of the original file
2. Looks up metadata on CivitAI by hash (model name, version, trigger words, etc.)
3. Fixes malformed safetensors headers (strips `__metadata__` block and `model.diffusion_model.` key prefix)
4. Detects the model version (Flux 2 Klein 9B, SDXL, Pony, etc.)
5. Converts to `.ckpt` via LoRAConverter
6. Registers in `custom_lora.json` with name, version, trigger words, and CivitAI provenance
7. Propagates all metadata as macOS extended attributes onto the `.ckpt`

## Requirements

- **macOS** (uses extended attributes via `libc`)
- **Python 3.9+** (stdlib only, no pip installs)
- **LoRAConverter** on `$PATH` — built from the [Draw Things source](https://github.com/nicktids/draw-things-community). Pre-built signed binaries are included in `bin/`.

## Install

Copy the converter binaries to somewhere on your `$PATH`:

```bash
cp bin/LoRAConverter /usr/local/bin/
cp bin/ModelConverter /usr/local/bin/     # optional, for base models
cp bin/ModelQuantizer /usr/local/bin/     # optional, for quantization
cp bin/EmbeddingConverter /usr/local/bin/ # optional, for embeddings
```

The main script can run from anywhere:

```bash
# Option A: run directly
python3 dt-import.py my_lora.safetensors

# Option B: symlink into PATH
ln -s "$(pwd)/dt-import.py" /usr/local/bin/dt-import
dt-import my_lora.safetensors
```

## Quick start

**Import a single LoRA:**

```bash
python3 dt-import.py ~/Downloads/some_lora.safetensors
```

This writes the `.ckpt` to Draw Things' model directory and adds an entry to `custom_lora.json`. Restart Draw Things and the LoRA appears in your list.

**Import a folder of LoRAs:**

```bash
python3 dt-import.py ~/Downloads/loras/
```

Recursively finds all `.safetensors` files and processes each one.

**Preview without changing anything:**

```bash
python3 dt-import.py --dry-run ~/Downloads/some_lora.safetensors
```

## CLI reference

```
dt-import.py [OPTIONS] FILE_OR_DIR [FILE_OR_DIR ...]
```

### Options

| Flag | Description |
|---|---|
| `--output-dir DIR` | Where to write `.ckpt` files. Default: Draw Things model directory |
| `--json-file PATH` | Path to `custom_lora.json`. Default: `<output-dir>/custom_lora.json` |
| `--version VER` | Override model version for all files (e.g. `flux2_9b`, `sdxl_base_v0.9`) |
| `--scale-factor N` | Network scale factor passed to LoRAConverter |
| `--name NAME` | Override display name (single-file mode only) |
| `--size-limit MB` | Max input file size in MB. Default: 1200. Set to 0 for unlimited |
| `--dry-run` | Show what would happen without doing anything |
| `--non-interactive` | Skip files that need user input instead of prompting |

### Pipeline modes

These are mutually exclusive. Pick one or use the default (full pipeline).

| Flag | What runs |
|---|---|
| *(default)* | Full pipeline: SHA256 → CivitAI → fix → convert → register |
| `--fix-only` | SHA256 + CivitAI + header fix. No conversion, no JSON |
| `--tag-only` | SHA256 + CivitAI xattrs only. No header fix, no conversion |

### Step overrides

Skip individual stages within any mode.

| Flag | What it skips |
|---|---|
| `--skip-fix` | Header fix (file is already clean) |
| `--skip-civitai` | CivitAI API call (offline use, or private LoRA not on CivitAI) |
| `--skip-convert` | LoRAConverter (tag and fix only, stop before conversion) |

## Config file

Instead of passing flags every time, create a config file:

```bash
mkdir -p ~/.config/dt-model-helper
cat > ~/.config/dt-model-helper/config << 'EOF'
output-dir=/path/to/your/models
json-file=/path/to/custom_lora.json
size-limit=1500
non-interactive=true
EOF
```

One setting per line, `key=value`. Keys are the same as CLI flags without `--`. CLI flags always override config values.

The config file is looked up at `$XDG_CONFIG_HOME/dt-model-helper/config`, falling back to `~/.config/dt-model-helper/config`.

## Environment variables

| Variable | Purpose |
|---|---|
| `DT_OUTPUT_DIR` | Default output directory (overridden by `--output-dir` and config) |
| `DT_JSON_FILE` | Default JSON file path (overridden by `--json-file` and config) |

## How version detection works

The tool resolves the Draw Things model version in this priority order:

1. **`--version` flag** — always wins
2. **CivitAI `baseModel`** — authoritative, fetched automatically by file hash
3. **Safetensors key patterns** — fallback heuristic for files not on CivitAI
4. **Interactive prompt** — asks you to pick (or skips the file in `--non-interactive` mode)

Supported versions: `flux2`, `flux2_9b`, `flux2_4b`, `flux1`, `z_image`, `qwen_image`, `sdxl_base_v0.9`, `sd3`, `sd3_large`, `pixart`, `auraflow`, `v1`, `v2`

## JSON output format

Each imported LoRA gets an entry like this in `custom_lora.json`:

```json
{
  "sha256": "d6a6b38410a9b8fda5e8e180aac32cd731fd1596e73616ee60a2b5b476cbfbe2",
  "name": "Selfies i2i",
  "version": "flux2_9b",
  "file": "selfies_i2i_f16.ckpt",
  "prefix": "Turn this into a selfie. View from above. ",
  "is_lo_ha": false,
  "civitai": {
    "nsfwLevel": 3,
    "baseModel": "Flux.2 Klein 9B",
    "name": "Selfies (image2selfie) | Klein-9b i2i",
    "air": "urn:air:flux2:lora:civitai:2381375@2677942",
    "sha256": "fd8a64bd9df258a5fc720ae5c09e3c30e6d84dbd399c83c781c8fd3b853fd8bd"
  }
}
```

- `sha256` — SHA256 of the output `.ckpt` file
- `name` — cleaned display name (architecture tokens stripped since DT filters by version)
- `prefix` — trigger words. Period-separated for Flux/T5 models, comma-separated for SDXL/CLIP models
- `civitai.sha256` — SHA256 of the original `.safetensors` before any header fixes
- `civitai.name` — original unmodified name from CivitAI
- `civitai.air` — CivitAI resource identifier

## Name cleanup

The display name is derived from the CivitAI model name (or filename as fallback). The tool automatically:

- Removes parenthesized content: `Selfies (image2selfie)` → `Selfies`
- Strips architecture tokens: `Klein-9b`, `SDXL`, `Flux`, `LoRA`, etc. (redundant since DT shows compatible models)
- Applies base model prefixes: Pony → `PD-`, Illustrious → `IL-`, SDXL 1.0 → `XL-`
- Qwen special handling: moves version numbers (2509/2511/2512) to front of name

## Metadata (xattrs)

All CivitAI metadata is stored as macOS extended attributes on both the source `.safetensors` and output `.ckpt` files. This means you can always trace where a model came from:

```bash
xattr -l my_model.ckpt
```

Key attributes:

| xattr | Content |
|---|---|
| `civitai.sha256sum` | SHA256 of the original safetensors file |
| `civitai.baseModel` | CivitAI base model (e.g. "Flux.2 Klein 9B") |
| `civitai.name` | CivitAI model name |
| `civitai.air` | CivitAI resource identifier |
| `civitai.triggerWords` | JSON array of trigger words |
| `civitai.metadata.version` | Metadata schema version (currently `1`) |
| `com.apple.metadata:kMDItemWhereFroms` | CivitAI model page URL (Spotlight-indexed) |

## Duplicate detection

Re-running the tool on the same file won't create duplicates. It checks:

1. CivitAI SHA256 match in existing JSON entries
2. `.ckpt` filename match
3. CivitAI model version ID match (from `air` field)

## Backups

Before modifying `custom_lora.json`, the tool backs up the current version to `~/.cache/kernkraft235/`. Up to 10 backups are kept, oldest pruned automatically.

## Other binaries

The `bin/` directory also includes signed converters for other Draw Things model types:

| Binary | Purpose |
|---|---|
| `ModelConverter` | Convert base model checkpoints |
| `ModelQuantizer` | Quantize models (reduce file size) |
| `EmbeddingConverter` | Convert textual inversion embeddings |

These are built from the Draw Things source and work the same way — pass `--help` to any of them for usage.
