# TURNOVER / SPEC / ROUGH PLAN

**Project: safetensors → Draw Things Importer** _Turnover spec — February 2026_

---

## Current Working LoRA Flow (Implemented)

`import-lora.py` now automates the LoRA path you described:

1. Runs `fix-safetensors-header.py --overwrite` on each input `.safetensors`.
2. Calls Draw Things `LoRAConverter` for actual safetensors → `.ckpt`
   conversion.
3. Copies provenance xattrs from source safetensors to output `.ckpt`:
   - `kernkraft235.safetensors.pre-sha256sum`
   - `kernkraft235.safetensors.fix-v1`
   - all `kernkraft235.civitai.*`
   - `com.apple.metadata:kMDItemWhereFroms`
4. Upserts a matching entry into `custom_lora.json`.

The converter step intentionally uses Draw Things' own converter logic rather
than re-implementing `LoRAImporter.swift` in Python.

### Usage

Using a built converter binary:

```bash
./import-lora.py \
  --output-dir "/path/to/DrawThings/Models" \
  --custom-lora-json "/path/to/DrawThings/Models/custom_lora.json" \
  --converter-bin "/path/to/LoRAConverter" \
  "/path/to/your_lora.safetensors"
```

Using a local `draw-things-community` repo with Bazel:

```bash
./import-lora.py \
  --output-dir "/path/to/DrawThings/Models" \
  --custom-lora-json "/path/to/DrawThings/Models/custom_lora.json" \
  --converter-repo "/path/to/draw-things-community" \
  "/path/to/your_lora.safetensors"
```

If this script sits next to a cloned `./draw-things-community`,
`--converter-repo` is auto-detected.

```bash
./import-lora.py \
  --output-dir "/path/to/DrawThings/Models" \
  --custom-lora-json "/path/to/DrawThings/Models/custom_lora.json" \
  "/path/to/your_lora.safetensors"
```

Batch usage:

```bash
./import-lora.py \
  --output-dir "/path/to/DrawThings/Models" \
  --custom-lora-json "/path/to/DrawThings/Models/custom_lora.json" \
  --converter-bin "/path/to/LoRAConverter" \
  /path/to/loras/*.safetensors
```

Useful flags:

- `--skip-fix` if input files are already fixed/tagged.
- `--version <model_version>` to force converter version.
- `--scale-factor <float>` to pass LoRA scaling into converter.
- `--prefix "..."` to set the `prefix` field written into `custom_lora.json`.
- `--continue-on-error` to process remaining files if one fails.

Bazel note: this repo pins a specific Bazel version via `.bazelversion`
(currently `7.4.1`). If your `bazel` is a different version, use Bazelisk:

```bash
./import-lora.py ... --converter-repo ./draw-things-community --bazel-bin bazelisk ...
```

---

**Background & Prior Work**

A previous script (`fix-safetensors-header.py`) handles safetensors
preprocessing and metadata tagging. The new script builds on the infrastructure
established there but is intentionally separate — it is a deliberate,
manual-invoke tool, not a folder action.

---

**What fix-safetensors-header.py already does (context for new work)**

Operates on `.safetensors` files. Performs two independent header fixes, tracked
with a bitfield xattr:

- Strips `__metadata__` block from the safetensors header JSON (bit 1)
- Strips `model.diffusion_model.` prefix from tensor key names (bit 2)

Writes the following xattrs to the file before any modification occurs:

| xattr key                                | value                  | notes                                                                                                                           |
| ---------------------------------------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `kernkraft235.safetensors.pre-sha256sum` | hex string             | SHA-256 of original unmodified file. Written first, before any changes. If `fix-v1 > 0` and this is absent, it's unrecoverable. |
| `kernkraft235.safetensors.fix-v1`        | decimal bitmask string | 0=clean, 1=metadata stripped, 2=keys renamed, 3=both                                                                            |
| `kernkraft235.civitai.id`                | string                 | CivitAI version ID                                                                                                              |
| `kernkraft235.civitai.modelId`           | string                 | CivitAI model ID                                                                                                                |
| `kernkraft235.civitai.modelType`         | string                 | e.g. `LORA`, `Checkpoint`                                                                                                       |
| `kernkraft235.civitai.baseModel`         | string                 | e.g. `Flux.1 D`, `SD 1.5`                                                                                                       |
| `kernkraft235.civitai.baseModel`         | string                 | e.g. `Flux.1 D`                                                                                                                 |
| `kernkraft235.civitai.nsfwLevel`         | string                 | numeric                                                                                                                         |
| `kernkraft235.civitai.triggerWords`      | JSON array string      | from `trainedWords`                                                                                                             |
| `kernkraft235.civitai.name`              | string                 | model name from `model.name`                                                                                                    |
| `kernkraft235.civitai.air`               | string                 | AIR identifier                                                                                                                  |
| `kernkraft235.civitai.fileName`          | string                 | original filename from `files[0].name`                                                                                          |
| `kernkraft235.civitai.description`       | string                 |                                                                                                                                 |
| `kernkraft235.civitai.images`            | JSON array string      | list of image URLs                                                                                                              |
| `com.apple.metadata:kMDItemWhereFroms`   | binary plist array     | set to `["https://civitai.com/models/{modelId}"]`                                                                               |

xattr helper layer uses `ctypes` against macOS `libc` directly
(`getxattr`/`setxattr` with the two trailing macOS-specific `0, 0` int args). No
third-party packages anywhere in this stack.

---

**New Script: Draw Things Importer**

**Purpose:** Convert `.safetensors` model files into Draw Things' native
checkpoint format (SQLite-based), register them in Draw Things' model index
JSON, and carry metadata forward so provenance is never lost.

**Invoke:** Manual only. Not a folder action. Not run by Automator or launchd.

**Inputs:** One or more `.safetensors` files (already processed by
`fix-safetensors-header.py` ideally, but not required).

---

**Step 1 — Conversion**

Draw Things stores models as a proprietary SQLite database rather than
safetensors JSON+binary. The conversion process was previously
reverse-engineered. A reference document and/or prior manual conversion of a
text encoder exists and should be used as the primary source of truth for the
conversion logic. The Draw Things GitHub repo may contain additional hints on
the import format.

Key known facts:

- The app normally converts on import, one model at a time, through the UI
- Text encoder conversion was already done outside the app successfully (use
  this as the reference implementation)
- The app has no facility to convert text encoders at all — this was solved
  manually

**Task:** Implement the conversion from `.safetensors` → Draw Things checkpoint
format (SQLite) in Python, based on the reference document and repo analysis.
Must be able to batch process multiple files.

---

**Step 2 — Register in Draw Things model index**

After conversion, Draw Things needs to be made aware of the new file. There is a
JSON index file that the app reads to enumerate available models. The new
checkpoint must be inserted into this file with the correct fields. The exact
schema of this JSON needs to be confirmed from the repo or by inspecting an
existing entry, but it is a required step for Draw Things to surface the model.

---

**Step 3 — Metadata propagation to checkpoint file**

After a successful conversion, copy the following from the source `.safetensors`
file's xattrs onto the output checkpoint file:

- `kernkraft235.safetensors.pre-sha256sum` → carry forward verbatim. **This is
  the original pre-modification SHA-256 of the source safetensors file.** It is
  not a hash of the checkpoint. It is preserved purely for CivitAI API lookups
  and provenance. Never recompute it on the checkpoint.
- All `kernkraft235.civitai.*` xattrs → carry forward verbatim
- `kernkraft235.safetensors.fix-v1` → carry forward verbatim (records what state
  the source was in)
- `com.apple.metadata:kMDItemWhereFroms` → carry forward (the CivitAI model page
  URL)

The intent is that the checkpoint file is fully self-describing with respect to
its origin, even though its own SHA-256 would be meaningless for CivitAI
lookups.

---

**Constraints & Notes**

- No third-party Python packages unless unavoidable. stdlib preferred
  throughout.
- Fish shell is the default shell environment; scripts should use shebangs and
  be directly executable.
- Target machine is macOS (M3 Max, Apple Silicon). xattr syscall signatures are
  macOS-specific.
- The fix script's xattr layer (`ctypes`/`libc`) can be copied or imported as a
  shared module if both scripts end up in the same directory.
- The sha256 carried to the checkpoint is always the _pre-modification original_
  — never a new hash. This is the whole point.

---

# FOLDER CONTENTS SO FAR

src/ The folder with parts of the Drawthings community repo that might be
helpful importers/ Symlinks to some of `src/Libraries/ModelOp/*` Converters/
Symlinks to `src/Apps/*Converter` src/Libraries/ModelZoo Was unsure how
important this was but its another library that _might_ be relevant.

references/ old files that can help guide you
references/flux_2_klein_4b_q6p.ckpt sqlite file showing how Draw Things stores
its data, this file is separate from the .ckpt-tensordata file that actually
contains the tensors. This is not a normal set up, its for when models are
selected to "speed up loading", but the fact that it splits off the sqlite db
was advantageous to show you the structure references/safe_swap.py script I used
last week to successfully convert a text encoder safetensors to ckpt. It has
hardcoded paths which is yuck, and should only serve as a basis. IIRC the
`INPUT_DONOR_CKPT` wasn’t required but idk references/custom.json the json file
format that drawthings stores model data in, this is how the app know it has
models to use references/custom_lora.json same thing but for loras

both the jsons have been pruned because they are really much longer but its not
necessary for you to see all that just to see the structure
