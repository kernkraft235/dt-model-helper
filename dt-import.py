#!/usr/bin/env python3
"""
Draw Things LoRA unified import pipeline.

Full pipeline per file:
  1. SHA256        — compute or read from xattr
  2. CivitAI       — fetch by hash if xattrs missing
  3. Header fix    — strip __metadata__ + model.diffusion_model. prefix
  4. Pre-flight    — size check, LoRA type check, duplicate check
  5. Version       — resolve from CivitAI baseModel (authoritative)
  6. Convert       — LoRAConverter with --version always set
  7. Register      — JSON entry with civitai sub-object
  8. Propagate     — xattrs → .ckpt file

Usage:
    dt-import.py [OPTIONS] <file_or_dir> [file_or_dir ...]
    dt-import.py --fix-only <file_or_dir> [file_or_dir ...]
    dt-import.py --tag-only <file_or_dir> [file_or_dir ...]

Requires: LoRAConverter on $PATH (built from Draw Things source).
macOS only (xattr support via ctypes against libc).
Python 3.9+ stdlib only — no third-party dependencies.
"""
from __future__ import annotations

import sys
import os
import re
import json
import struct
import shutil
import hashlib
import plistlib
import argparse
import subprocess
import urllib.request
import urllib.error
import ctypes
import ctypes.util
from pathlib import Path
from datetime import datetime

# ── constants ────────────────────────────────────────────────────────────────

DEFAULT_OUTPUT_DIR = Path.home() / "Library/Containers/com.liuliu.draw-things/Data/Documents/Models"
BACKUP_DIR = Path.home() / ".cache/kernkraft235"
MAX_BACKUPS = 10
DEFAULT_SIZE_LIMIT_MB = 1200

CIVITAI_PFX = "civitai."
XATTR_SHA256 = b"civitai.sha256sum"
XATTR_FIX = b"kernkraft235.safetensors.fix-v1"
XATTR_WHERE = b"com.apple.metadata:kMDItemWhereFroms"

# CivitAI baseModel → Draw Things version string
CIVITAI_TO_DT_VERSION = {
    "ZImageTurbo":      "z_image",
    "Flux.2 Klein 9B":  "flux2_9b",
    "Flux.2 Klein 4B":  "flux2_4b",
    "Pony":             "sdxl_base_v0.9",
    "Illustrious":      "sdxl_base_v0.9",
    "SDXL 1.0":         "sdxl_base_v0.9",
    "Qwen":             "qwen_image",
}

# CivitAI baseModel → name prefix
CIVITAI_NAME_PREFIX = {
    "Pony":         "PD-",
    "Illustrious":  "IL-",
    "SDXL 1.0":     "XL-",
}

# Known DT version strings for interactive prompts
DT_VERSIONS = [
    "z_image", "flux2_9b", "flux2_4b", "flux2", "qwen_image",
    "sdxl_base_v0.9", "sd3", "sd3_large", "pixart", "auraflow",
    "flux1", "v1", "v2",
]

# CivitAI API field extractors (from API response → xattr value)
# Only fields we actually use: version resolution, name, trigger words, JSON embed
CIVITAI_FIELD_EXTRACTORS = {
    "nsfwLevel":    lambda d: d.get("nsfwLevel"),
    "baseModel":    lambda d: d.get("baseModel"),
    "triggerWords": lambda d: d.get("trainedWords"),      # renamed
    "name":         lambda d: d.get("model", {}).get("name"),
    "modelType":    lambda d: d.get("model", {}).get("type"),
    "air":          lambda d: d.get("air"),
    "modelId":      lambda d: d.get("modelId"),           # for WhereFroms URL only
}

# Field names for reading xattrs back (same keys as extractors)
CIVITAI_FIELDS = list(CIVITAI_FIELD_EXTRACTORS.keys())

# Header fix constants
HDR_PREFIX = "model.diffusion_model."
FIX_METADATA = 1
FIX_KEYS = 2

# CivitAI API
CIVITAI_BY_HASH = "https://civitai.com/api/v1/model-versions/by-hash/{}"
CIVITAI_MODEL_URL = "https://civitai.com/models/{}"


# ── xattr helpers (ctypes against macOS libc) ────────────────────────────────

_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)


def _xattr_get(path: str, name: bytes) -> bytes | None:
    path_b = path.encode() if isinstance(path, str) else path
    size = _libc.getxattr(path_b, name, None, 0, 0, 0)
    if size < 0:
        return None
    buf = ctypes.create_string_buffer(size)
    ret = _libc.getxattr(path_b, name, buf, size, 0, 0)
    if ret < 0:
        return None
    return buf.raw[:ret]


def _xattr_set(path: str, name: bytes, value: bytes) -> None:
    path_b = path.encode() if isinstance(path, str) else path
    ret = _libc.setxattr(path_b, name, value, len(value), 0, 0)
    if ret != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno), path)


def _xattr_list(path: str) -> list[bytes]:
    path_b = path.encode() if isinstance(path, str) else path
    size = _libc.listxattr(path_b, None, 0, 0)
    if size <= 0:
        return []
    buf = ctypes.create_string_buffer(size)
    ret = _libc.listxattr(path_b, buf, size, 0)
    if ret <= 0:
        return []
    return [x for x in buf.raw[:ret].split(b"\x00") if x]


def get_civitai_xattr(path: str, field: str) -> str | None:
    raw = _xattr_get(path, (CIVITAI_PFX + field).encode())
    if raw is None:
        return None
    return raw.decode("utf-8", errors="replace")


def get_sha256_xattr(path: str) -> str | None:
    raw = _xattr_get(path, XATTR_SHA256)
    if raw is None:
        return None
    return raw.decode("utf-8", errors="replace")


def get_fix_xattr(path: str) -> int | None:
    raw = _xattr_get(path, XATTR_FIX)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def civitai_xattr_exists(path: str) -> bool:
    """True if any civitai.* data key is already set."""
    return _xattr_get(path, (CIVITAI_PFX + "baseModel").encode()) is not None


def mark_sha256(path: str, hexdigest: str) -> None:
    _xattr_set(path, XATTR_SHA256, hexdigest.encode())


def mark_fix(path: str, fix_flags: int) -> None:
    _xattr_set(path, XATTR_FIX, str(fix_flags).encode())


def mark_civitai(path: str, data: dict) -> int:
    """Write extracted CivitAI fields as individual xattrs. Returns count written."""
    written = 0
    for field, extractor in CIVITAI_FIELD_EXTRACTORS.items():
        value = extractor(data)
        if value is None:
            continue
        if isinstance(value, (list, dict)):
            raw = json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()
        else:
            raw = str(value).encode()
        _xattr_set(path, (CIVITAI_PFX + field).encode(), raw)
        written += 1
    # Metadata version stamp
    _xattr_set(path, (CIVITAI_PFX + "metadata.version").encode(), b"1")
    return written


def mark_where_froms(path: str, url: str) -> None:
    """Overwrite kMDItemWhereFroms with a single-element binary plist array."""
    plist_bytes = plistlib.dumps([url], fmt=plistlib.FMT_BINARY)
    _xattr_set(path, XATTR_WHERE, plist_bytes)


def propagate_xattrs(src: str, dst: str) -> int:
    """Copy all civitai.* and kMDItemWhereFroms xattrs from src to dst."""
    copied = 0
    for name in _xattr_list(src):
        name_str = name.decode("utf-8", errors="replace")
        if name_str.startswith(CIVITAI_PFX) or name == XATTR_WHERE:
            val = _xattr_get(src, name)
            if val is not None:
                try:
                    _xattr_set(dst, name, val)
                    copied += 1
                except OSError:
                    pass
    return copied


# ── SHA256 ───────────────────────────────────────────────────────────────────

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(16 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def ensure_sha256(src_path: str) -> str:
    """Get SHA256 — from xattr if available, else compute and tag."""
    existing = get_sha256_xattr(src_path)
    if existing:
        return existing
    digest = sha256_file(src_path)
    mark_sha256(src_path, digest)
    return digest


# ── CivitAI API ──────────────────────────────────────────────────────────────

def fetch_civitai(sha256: str) -> dict | None:
    url = CIVITAI_BY_HASH.format(sha256.upper())
    req = urllib.request.Request(url, headers={"User-Agent": "dt-import/2.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    except Exception:
        return None


def ensure_civitai_xattrs(src_path: str, sha256: str) -> bool:
    """
    Fetch CivitAI data by hash if xattrs not already present.
    Returns True if CivitAI data is available (pre-existing or freshly fetched).
    """
    if civitai_xattr_exists(src_path):
        return True

    data = fetch_civitai(sha256)
    if data is None:
        return False

    n = mark_civitai(src_path, data)
    model_id = data.get("modelId")
    if model_id is not None:
        model_url = CIVITAI_MODEL_URL.format(model_id)
        mark_where_froms(src_path, model_url)
        print(f"  [civitai] {n} fields fetched, WhereFroms → {model_url}")
    else:
        print(f"  [civitai] {n} fields fetched")
    return True


# ── safetensors header parsing & fixing ──────────────────────────────────────

def read_safetensors_header(path: str) -> dict | None:
    """Read the JSON header from a safetensors file (no tensor data)."""
    try:
        with open(path, "rb") as f:
            raw_len = f.read(8)
            if len(raw_len) < 8:
                return None
            header_len = struct.unpack("<Q", raw_len)[0]
            if header_len > 100 * 1024 * 1024:  # sanity: 100MB header max
                return None
            return json.loads(f.read(header_len).decode("utf-8"))
    except Exception:
        return None


def detect_fixes(header: dict) -> int:
    """Detect what header fixes are needed. Returns bitmask."""
    flags = 0
    if "__metadata__" in header:
        flags |= FIX_METADATA
    if any(k.startswith(HDR_PREFIX) for k in header):
        flags |= FIX_KEYS
    return flags


def build_fixed_header(header: dict, flags: int) -> bytes:
    """Build the fixed header bytes."""
    if flags & FIX_METADATA:
        header.pop("__metadata__", None)
    new_header = {}
    for k, v in header.items():
        if (flags & FIX_KEYS) and k.startswith(HDR_PREFIX):
            new_header[k[len(HDR_PREFIX):]] = v
        else:
            new_header[k] = v
    return json.dumps(new_header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def fix_header_inplace(src_path: str, sha256: str) -> None:
    """
    Fix safetensors header in-place if needed.
    Sets fix-v1 xattr. Re-applies sha256 xattr after overwrite.
    """
    fix_flags = get_fix_xattr(src_path)

    # Already processed
    if fix_flags is not None:
        if fix_flags > 0:
            fix_desc = "+".join(
                n for bit, n in ((FIX_METADATA, "metadata"), (FIX_KEYS, "keys"))
                if fix_flags & bit
            )
            print(f"  [fix] already fixed (xattr={fix_flags}: {fix_desc})")
        else:
            print(f"  [fix] no fix needed (clean)")
        return

    p = Path(src_path)
    with p.open("rb") as fin:
        header_len = struct.unpack("<Q", fin.read(8))[0]
        header = json.loads(fin.read(header_len).decode("utf-8"))
        flags = detect_fixes(header)

        if not flags:
            mark_fix(src_path, 0)
            print(f"  [fix] no fix needed (clean)")
            return

        fixed_bytes = build_fixed_header(header, flags)
        tmp = p.with_suffix(".tmp.safetensors")

        with tmp.open("wb") as fout:
            fout.write(struct.pack("<Q", len(fixed_bytes)))
            fout.write(fixed_bytes)
            while True:
                chunk = fin.read(16 * 1024 * 1024)
                if not chunk:
                    break
                fout.write(chunk)

    tmp.replace(p)
    mark_sha256(src_path, sha256)  # re-apply — overwrite lost the xattr
    mark_fix(src_path, flags)

    fix_desc = "+".join(
        n for bit, n in ((FIX_METADATA, "metadata"), (FIX_KEYS, "keys"))
        if flags & bit
    )
    print(f"  [fix] header fixed (xattr={flags}: {fix_desc})")


# ── version detection from safetensors keys ──────────────────────────────────

def detect_version_from_keys(header: dict) -> str | None:
    """
    Detect LoRA model version from safetensors tensor key names.
    Fallback for files without CivitAI data.
    """
    keys = set(header.keys()) - {"__metadata__"}
    key_str = " ".join(keys)

    # Heuristic 0: __metadata__.modelspec.architecture
    metadata = header.get("__metadata__", {})
    if isinstance(metadata, dict):
        arch = metadata.get("modelspec.architecture", "")
        if arch:
            arch_lower = arch.lower()
            if "flux" in arch_lower:
                pass  # fall through to block counting
            elif "sdxl" in arch_lower:
                return "sdxl_base_v0.9"
            elif "z_image" in arch_lower or "zimagetturbo" in arch_lower:
                return "z_image"
            elif "qwen" in arch_lower:
                return "qwen_image"

    # Heuristic 1: single_blocks check (LoRAImporter.swift-style)
    is_flux2 = ("single_blocks.39.linear1" in key_str
                or "single_blocks_39_linear1" in key_str
                or "single_transformer_blocks.39." in key_str
                or "single_transformer_blocks_39_" in key_str)
    if is_flux2:
        return "flux2"

    is_flux2_9b = ("single_blocks.23.linear1" in key_str
                   or "single_blocks_23_linear1" in key_str
                   or "single_transformer_blocks.23." in key_str
                   or "single_transformer_blocks_23_" in key_str)
    if is_flux2_9b:
        return "flux2_9b"

    is_flux2_4b = ("single_blocks.19.linear1" in key_str
                   or "single_blocks_19_linear1" in key_str
                   or "single_transformer_blocks.19." in key_str
                   or "single_transformer_blocks_19_" in key_str)
    if is_flux2_4b:
        return "flux2_4b"

    # Heuristic 2: OneTrainer/diffusers transformer_blocks range counting
    max_tb_idx = -1
    has_double_stream_mod = False
    tb_pattern = re.compile(r'(?:^|[._])transformer_blocks[._](\d+)[._]')
    for k in keys:
        m = tb_pattern.search(k)
        if m:
            max_tb_idx = max(max_tb_idx, int(m.group(1)))
        if "double_stream_modulation" in k:
            has_double_stream_mod = True

    if max_tb_idx >= 0 or has_double_stream_mod:
        is_flux_family = (has_double_stream_mod
                          or "flux" in metadata.get("modelspec.architecture", "").lower())

        if is_flux_family or max_tb_idx >= 19:
            if max_tb_idx >= 38:
                return "flux2"
            elif max_tb_idx >= 24:
                return "flux2"
            elif max_tb_idx >= 20:
                return "flux2_9b"
            else:
                return "flux2_4b"

        if max_tb_idx >= 59:
            return "qwen_image"

    # Z-Image
    if ("layers.29.feed_forward.w3." in key_str
            or "layers_29_feed_forward_w3" in key_str):
        return "z_image"

    # Qwen Image
    if ("transformer_blocks.59.txt_mlp." in key_str
            or "transformer_blocks_59_txt_mlp" in key_str
            or "transformer_blocks.37.attn." in key_str
            or "transformer_blocks_37_attn_" in key_str):
        return "qwen_image"

    # SDXL
    for k, v in header.items():
        if k == "__metadata__":
            continue
        k_flat = k.replace(".", "_")
        if ("input_blocks_4_1_transformer_blocks_0_attn2_to_k" in k_flat
                or "input_blocks.4.1.transformer_blocks.0.attn2.to_k" in k):
            shape = v.get("shape", [])
            if shape and shape[-1] == 2048:
                return "sdxl_base_v0.9"

    return None


def detect_is_loha(header: dict) -> bool:
    """Check if the LoRA uses LoHa (Hadamard) format."""
    for k in header:
        if k == "__metadata__":
            continue
        if ".hada_w1_a" in k or ".hada_w1_b" in k or "_hada_w1_a" in k or "_hada_w1_b" in k:
            return True
    return False


# ── version resolution ───────────────────────────────────────────────────────

def resolve_version(
    src_path: str,
    header: dict | None,
    cli_version: str | None,
    non_interactive: bool,
) -> str | None:
    """
    Resolve DT version. Priority:
      1. CLI --version override
      2. CivitAI baseModel xattr (authoritative after auto-fetch)
      3. Safetensors key-pattern detection (fallback for non-CivitAI files)
      4. Interactive prompt (or None if --non-interactive)
    """
    if cli_version:
        return cli_version

    base_model = get_civitai_xattr(src_path, "baseModel")
    if base_model and base_model in CIVITAI_TO_DT_VERSION:
        return CIVITAI_TO_DT_VERSION[base_model]

    if header:
        detected = detect_version_from_keys(header)
        if detected:
            return detected

    return None


def prompt_version(filename: str, non_interactive: bool) -> str | None:
    """Prompt user for version selection, or return None in non-interactive mode."""
    if non_interactive:
        return None
    print(f"\n  Cannot auto-detect version for: {filename}")
    print("  Select model version:")
    for i, v in enumerate(DT_VERSIONS, 1):
        print(f"    {i:2d}. {v}")
    print(f"    {len(DT_VERSIONS)+1:2d}. [skip this file]")
    while True:
        try:
            choice = input("  Choice: ").strip()
            idx = int(choice)
            if 1 <= idx <= len(DT_VERSIONS):
                return DT_VERSIONS[idx - 1]
            if idx == len(DT_VERSIONS) + 1:
                return None
        except (ValueError, EOFError, KeyboardInterrupt):
            return None
        print("  Invalid choice, try again.")


# ── name derivation ──────────────────────────────────────────────────────────

# Architecture/version tokens to strip from display names (case-insensitive).
# DT already shows compatible models, so these are redundant in the name.
_NAME_STRIP_TOKENS = [
    "Klein-9b", "Klein-4b", "Klein 9b", "Klein 4b",
    "SDXL", "SD 1.5", "SD1.5", "SD 2.1", "SD2.1",
    "Flux.2", "Flux.1", "Flux 2", "Flux 1", "Flux",
    "Pony", "Illustrious", "Z-Image",
    "LoRA", "LoHa", "LoCon",
    "9B", "4B",
]
_NAME_STRIP_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(t) for t in _NAME_STRIP_TOKENS) + r")\b",
    re.IGNORECASE,
)


def derive_name(
    src_path: str,
    cli_name: str | None,
    civitai_base_model: str | None,
) -> str:
    """
    Derive display name for the LoRA.
    Priority: --name > CivitAI xattr name > filename stem
    Then apply prefix and cleanup rules.
    """
    if cli_name:
        raw_name = cli_name
    else:
        civitai_name = get_civitai_xattr(src_path, "name")
        if civitai_name:
            raw_name = civitai_name
        else:
            raw_name = Path(src_path).stem

    # Remove parenthesized content
    raw_name = re.sub(r"\([^)]*\)", "", raw_name).strip()

    # Strip architecture/version tokens (DT filters by version already)
    raw_name = _NAME_STRIP_RE.sub("", raw_name)

    # Collapse whitespace first so pipe checks work cleanly
    raw_name = re.sub(r"\s+", " ", raw_name).strip()
    # Remove orphan pipes: "Foo | " or " | Bar" or bare " | "
    raw_name = re.sub(r"\s*\|\s*$", "", raw_name)
    raw_name = re.sub(r"^\s*\|\s*", "", raw_name)
    # Remove pipe between segments where one side got stripped to nothing
    # e.g. "Selfies |  i2i" → "Selfies i2i"
    raw_name = re.sub(r"\s*\|\s*", " ", raw_name)
    # Clean up leading/trailing junk
    raw_name = re.sub(r"^[\s\-_|]+", "", raw_name)
    raw_name = re.sub(r"[\s\-_|]+$", "", raw_name)
    raw_name = re.sub(r"\s+", " ", raw_name).strip()

    # Qwen special: if name contains 2509/2511/2512, move to front
    if civitai_base_model == "Qwen":
        for num in ("2512", "2511", "2509"):
            if num in raw_name:
                cleaned = raw_name.replace(num, "").strip(" -_")
                cleaned = re.sub(r"\s+", " ", cleaned).strip()
                raw_name = f"{num}-{cleaned}" if cleaned else num
                break

    # Apply base model prefix
    if civitai_base_model and civitai_base_model in CIVITAI_NAME_PREFIX:
        prefix = CIVITAI_NAME_PREFIX[civitai_base_model]
        if not raw_name.startswith(prefix):
            raw_name = prefix + raw_name

    return raw_name


# ── trigger words ────────────────────────────────────────────────────────────

# Versions using CLIP text encoder (comma-separated trigger words)
_CLIP_VERSIONS = {"sdxl_base_v0.9", "v1", "v2"}
# Versions using T5/LLM text encoder (period-separated trigger words)
_T5_VERSIONS = {"flux2", "flux2_9b", "flux2_4b", "flux1", "qwen_image", "z_image"}


def derive_trigger_words(src_path: str, dt_version: str | None) -> str:
    """
    Build the 'prefix' field from CivitAI triggerWords xattr.
    Separator: commas for CLIP-based models, periods for T5/LLM-based.
    """
    raw = get_civitai_xattr(src_path, "triggerWords")
    if not raw:
        return ""

    try:
        words = json.loads(raw)
        if not isinstance(words, list) or not words:
            return ""
    except (json.JSONDecodeError, TypeError):
        words = [raw]

    words = [str(w).strip() for w in words if w]
    if not words:
        return ""

    # Pick separator based on text encoder type
    use_periods = dt_version in _T5_VERSIONS

    if use_periods:
        # Period-separated: ensure each word ends with period, join with space
        parts = []
        for w in words:
            w = w.rstrip(",").strip()
            if not w.endswith("."):
                w += "."
            parts.append(w)
        joined = " ".join(parts)
        return joined + " "
    else:
        # Comma-separated
        parts = [w.rstrip(",.").strip() for w in words]
        joined = ", ".join(parts)
        return joined + ", "


# ── civitai data collection ──────────────────────────────────────────────────

# Fields to include in the JSON civitai sub-object (slim set)
_CIVITAI_JSON_FIELDS = ["nsfwLevel", "baseModel", "name", "air"]


def collect_civitai_data(src_path: str) -> dict | None:
    """
    Collect CivitAI xattr data into a slim dict for embedding in JSON.
    Only: nsfwLevel, baseModel, name, air, sha256 (pre-modification hash).
    """
    data = {}
    for field in _CIVITAI_JSON_FIELDS:
        val = get_civitai_xattr(src_path, field)
        if val is None:
            continue
        try:
            parsed = json.loads(val)
            data[field] = parsed
        except (json.JSONDecodeError, TypeError):
            data[field] = val

    sha256 = get_sha256_xattr(src_path)
    if sha256:
        data["sha256"] = sha256

    return data if data else None


# ── JSON backup ──────────────────────────────────────────────────────────────

def backup_json(json_path: Path) -> None:
    """Backup JSON file to ~/.cache/kernkraft235/, keeping max 10 backups."""
    if not json_path.exists():
        return

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{json_path.name}.{ts}"
    backup_path = BACKUP_DIR / backup_name
    shutil.copy2(str(json_path), str(backup_path))

    # Prune old backups
    prefix = json_path.name + "."
    backups = sorted(
        [f for f in BACKUP_DIR.iterdir() if f.name.startswith(prefix)],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    for old in backups[MAX_BACKUPS:]:
        old.unlink(missing_ok=True)


# ── JSON registration ────────────────────────────────────────────────────────

def load_json(json_path: Path) -> list:
    if not json_path.exists():
        return []
    try:
        return json.loads(json_path.read_text("utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def save_json(json_path: Path, data: list) -> None:
    json_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def is_duplicate(entries: list, ckpt_filename: str, sha256: str | None) -> str | None:
    """
    Check if this LoRA is already registered.
    Returns the existing name if duplicate, None otherwise.
    """
    for entry in entries:
        if entry.get("file") == ckpt_filename:
            return entry.get("name", ckpt_filename)
        civitai = entry.get("civitai", {})
        if sha256 and civitai.get("sha256") == sha256:
            return entry.get("name", ckpt_filename)
    return None


# ── pre-flight checks ────────────────────────────────────────────────────────

def preflight(
    src_path: str,
    entries: list,
    size_limit_mb: int,
) -> tuple[bool, str]:
    """Run pre-flight checks. Returns (ok, reason)."""
    p = Path(src_path)

    # Size check
    file_size = p.stat().st_size
    limit_bytes = size_limit_mb * 1024 * 1024
    if size_limit_mb > 0 and file_size > limit_bytes:
        return False, f"file too large ({file_size / (1024**2):.0f} MB > {size_limit_mb} MB limit, use --size-limit)"

    # Is it a LoRA?
    model_type = get_civitai_xattr(src_path, "modelType")
    if model_type and model_type.upper() != "LORA":
        return False, f"not a LoRA (CivitAI modelType={model_type!r})"

    # Duplicate check by SHA256
    sha256 = get_sha256_xattr(src_path)
    if sha256:
        existing = is_duplicate(entries, "", sha256)
        if existing:
            return False, f"already imported as {existing!r} (SHA256 match)"

    # Duplicate check by CivitAI model version ID
    civitai_id = get_civitai_xattr(src_path, "id")
    if civitai_id:
        for entry in entries:
            civitai = entry.get("civitai", {})
            entry_id = str(civitai.get("id", ""))
            if entry_id == civitai_id:
                return False, f"already imported as {entry.get('name', '?')!r} (CivitAI id match)"

    return True, ""


# ── pipeline stages ──────────────────────────────────────────────────────────

def stage_sha256_and_civitai(src_path: str, skip_civitai: bool) -> str:
    """
    Stage 1-2: Compute SHA256, fetch CivitAI data.
    Returns the SHA256 hex digest.
    """
    sha256 = ensure_sha256(src_path)
    print(f"  [sha256] {sha256}")

    if not skip_civitai:
        if civitai_xattr_exists(src_path):
            print(f"  [civitai] xattrs already present")
        else:
            has_data = ensure_civitai_xattrs(src_path, sha256)
            if not has_data:
                print(f"  [civitai] not found on CivitAI (non-CivitAI file?)")

    return sha256


def stage_header_fix(src_path: str, sha256: str) -> None:
    """Stage 3: Fix safetensors header in-place."""
    fix_header_inplace(src_path, sha256)


# ── main processing ──────────────────────────────────────────────────────────

def process_file(
    src_path: str,
    output_dir: Path,
    json_path: Path,
    entries: list,
    cli_version: str | None,
    cli_name: str | None,
    scale_factor: str | None,
    non_interactive: bool,
    size_limit_mb: int,
    dry_run: bool,
    skip_fix: bool,
    skip_civitai: bool,
    skip_convert: bool,
    fix_only: bool,
    tag_only: bool,
) -> bool:
    """Process a single safetensors file. Returns True on success."""
    p = Path(src_path).resolve()
    src_path = str(p)
    print(f"\n{'─'*60}")
    print(f"  {p.name}")
    print(f"{'─'*60}")

    # ── Stage 1-2: SHA256 + CivitAI ─────────────────────────────────────
    sha256 = stage_sha256_and_civitai(src_path, skip_civitai)

    # ── Stage 3: Header fix ──────────────────────────────────────────────
    if not skip_fix and not tag_only:
        stage_header_fix(src_path, sha256)

    # Early exit for partial pipeline modes
    if tag_only:
        print(f"  [done] tag-only mode complete")
        return True

    if fix_only:
        print(f"  [done] fix-only mode complete")
        return True

    if skip_convert:
        print(f"  [done] skip-convert mode (no LoRAConverter call)")
        return True

    # ── Stage 4: Pre-flight checks ───────────────────────────────────────
    ok, reason = preflight(src_path, entries, size_limit_mb)
    if not ok:
        print(f"  [skip] {reason}")
        return False

    # ── Stage 5: Version resolution ──────────────────────────────────────
    header = read_safetensors_header(src_path)
    version = resolve_version(src_path, header, cli_version, non_interactive)

    # Get CivitAI baseModel for name prefix logic
    civitai_base_model = get_civitai_xattr(src_path, "baseModel")

    # Derive name
    name = derive_name(src_path, cli_name, civitai_base_model)

    # Derive trigger words
    prefix = derive_trigger_words(src_path, version)

    # Detect LoHa
    is_loha = detect_is_loha(header) if header else False

    # Collect CivitAI data for JSON
    civitai_data = collect_civitai_data(src_path)

    # Report
    print(f"  name:     {name}")
    print(f"  version:  {version or '(unknown)'}")
    print(f"  prefix:   {prefix!r}")
    print(f"  is_loha:  {is_loha}")
    if civitai_data:
        print(f"  civitai:  {len(civitai_data)} fields")

    if dry_run:
        print("  [dry-run] would convert and register")
        return True

    # ── Stage 6: Call LoRAConverter ───────────────────────────────────────

    # Snapshot output dir before conversion
    before = set(output_dir.glob("*.ckpt")) if output_dir.exists() else set()

    cmd = [
        "LoRAConverter",
        "--file", src_path,
        "--name", name,
        "--output-directory", str(output_dir),
    ]
    # Always pass --version when known — LoRAConverter's built-in detection
    # fails on OneTrainer/diffusers format LoRAs (common format).
    if version:
        cmd += ["--version", version]
    if scale_factor:
        cmd += ["--scale-factor", scale_factor]

    print(f"  [run] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        stderr = result.stderr.strip()
        print(f"  [error] LoRAConverter failed (exit {result.returncode})")
        if stderr:
            print(f"  {stderr}")

        # If version was not set, maybe that's why it failed — try prompting
        if not version:
            version = prompt_version(p.name, non_interactive)
            if version:
                cmd_retry = [
                    "LoRAConverter",
                    "--file", src_path,
                    "--name", name,
                    "--output-directory", str(output_dir),
                    "--version", version,
                ]
                if scale_factor:
                    cmd_retry += ["--scale-factor", scale_factor]
                print(f"  [retry] {' '.join(cmd_retry)}")
                result = subprocess.run(cmd_retry, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"  [error] LoRAConverter still failed (exit {result.returncode})")
                    if result.stderr.strip():
                        print(f"  {result.stderr.strip()}")
                    return False
            else:
                print("  [skip] no version selected")
                return False

    if result.stdout.strip():
        print(f"  [out] {result.stdout.strip()}")

    # Find new .ckpt file
    after = set(output_dir.glob("*.ckpt")) if output_dir.exists() else set()
    new_files = after - before
    if not new_files:
        print("  [error] no new .ckpt file found in output directory")
        return False

    ckpt_path = sorted(new_files, key=lambda f: f.stat().st_mtime)[-1]

    # Clean up .ckpt filename: collapse consecutive _, remove _lora
    clean_name = ckpt_path.name
    clean_name = clean_name.replace("_lora_", "_")
    clean_name = re.sub(r"_+", "_", clean_name)
    if clean_name != ckpt_path.name:
        new_path = ckpt_path.with_name(clean_name)
        ckpt_path.rename(new_path)
        ckpt_path = new_path

    ckpt_filename = ckpt_path.name
    print(f"  [ok] → {ckpt_filename}")

    # Compute SHA256 of the .ckpt file
    ckpt_sha256 = sha256_file(str(ckpt_path))
    print(f"  [sha256] ckpt {ckpt_sha256}")

    # ── Stage 7: Register in JSON ────────────────────────────────────────

    existing = is_duplicate(entries, ckpt_filename, None)
    if existing:
        print(f"  [skip] JSON entry already exists for {ckpt_filename!r}")
    else:
        entry = {
            "sha256": ckpt_sha256,
            "name": name,
            "version": version or "",
            "file": ckpt_filename,
            "prefix": prefix,
            "is_lo_ha": is_loha,
        }
        if civitai_data:
            entry["civitai"] = civitai_data
        entries.append(entry)
        print(f"  [json] registered as {name!r}")

    # ── Stage 8: Propagate xattrs ────────────────────────────────────────

    n = propagate_xattrs(src_path, str(ckpt_path))
    print(f"  [xattr] copied {n} attributes to {ckpt_filename}")

    return True


# ── config file loader ────────────────────────────────────────────────────────

CONFIG_DIRS = [
    Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")),
]
CONFIG_SUBPATH = "dt-model-helper/config"


def load_config() -> dict[str, str]:
    """
    Load config from $XDG_CONFIG_HOME/dt-model-helper/config
    (or ~/.config/dt-model-helper/config). Format: key=value, one per line.
    Keys are CLI flag names without '--' (e.g. 'output-dir=/path').
    Lines starting with '#' are comments. Returns dict of key→value.
    """
    for base in CONFIG_DIRS:
        cfg = base / CONFIG_SUBPATH
        if cfg.exists():
            break
    else:
        return {}

    result = {}
    for line in cfg.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip("'\"")
            if key:
                result[key] = val
    return result


# ── CLI entry point ──────────────────────────────────────────────────────────

def collect_safetensors(paths: list[str]) -> list[Path]:
    """Expand directories and collect all .safetensors files."""
    result = []
    for p_str in paths:
        p = Path(p_str)
        if p.is_dir():
            result.extend(sorted(p.rglob("*.safetensors")))
        elif p.is_file() and p.suffix == ".safetensors":
            result.append(p)
        else:
            print(f"[warn] skipping {p_str} (not a .safetensors file or directory)")
    return result


def _cfg_get(config: dict, key: str, default=None):
    """Get a config value, converting types appropriately."""
    val = config.get(key)
    if val is None:
        return default
    return val


def main():
    # Load config file (CLI flags override config values)
    config = load_config()

    parser = argparse.ArgumentParser(
        description="Draw Things LoRA unified import pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
pipeline modes (mutually exclusive):
  (default)       Full: SHA256 → CivitAI → fix → convert → register
  --fix-only      SHA256 + CivitAI + header fix (no conversion/JSON)
  --tag-only      SHA256 + CivitAI xattrs only (no fix/conversion)

pipeline step overrides:
  --skip-fix      Skip header fix (file already clean)
  --skip-civitai  Skip CivitAI API fetch (offline / private LoRA)
  --skip-convert  Skip LoRAConverter (tag + fix only, then stop)

config file:
  $XDG_CONFIG_HOME/dt-model-helper/config (or ~/.config/dt-model-helper/config)
  One setting per line, key=value (same names as CLI flags without '--').
  CLI flags always override config file values.
  Example:
    output-dir=/path/to/models
    json-file=/path/to/custom_lora.json
    size-limit=1500
    non-interactive=true
""",
    )
    parser.add_argument("inputs", nargs="+", metavar="FILE_OR_DIR")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--json-file", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--scale-factor", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--size-limit", type=int, default=None, metavar="MB",
                        help=f"Max file size in MB (default: {DEFAULT_SIZE_LIMIT_MB}, 0=unlimited)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--non-interactive", action="store_true")

  #  mode_group = parser.add_mutually_exclusive_group()
 #   mode_group.add_argument("--fix-only", action="store_true")
 #   mode_group.add_argument("--tag-only", action="store_true")
#
 #   parser.add_argument("--skip-fix", action="store_true")
 #   parser.add_argument("--skip-civitai", action="store_true")
  #  parser.add_argument("--skip-convert", action="store_true")

    args = parser.parse_args()

    # Resolve settings: CLI > config > env > defaults
    output_dir_str = (args.output_dir
                      or _cfg_get(config, "output-dir")
                      or os.environ.get("DT_OUTPUT_DIR")
                      or str(DEFAULT_OUTPUT_DIR))
    output_dir = Path(output_dir_str)

    json_file_str = (args.json_file
                     or _cfg_get(config, "json-file")
                     or os.environ.get("DT_JSON_FILE"))
    json_path = Path(json_file_str) if json_file_str else output_dir / "custom_lora.json"

    if args.size_limit is not None:
        size_limit_mb = args.size_limit
    elif "size-limit" in config:
        size_limit_mb = int(config["size-limit"])
    else:
        size_limit_mb = DEFAULT_SIZE_LIMIT_MB

    non_interactive = args.non_interactive or _cfg_get(config, "non-interactive") == "true"
    cli_version = args.version or _cfg_get(config, "version")
    scale_factor = args.scale_factor or _cfg_get(config, "scale-factor")

    needs_convert = not (args.fix_only or args.tag_only or args.skip_convert)

    # Collect files
    files = collect_safetensors(args.inputs)
    if not files:
        print("No .safetensors files found.")
        sys.exit(1)

    if args.name and len(files) > 1:
        print("[error] --name can only be used with a single file")
        sys.exit(1)

    # Verify output dir and LoRAConverter only if we'll be converting
    if needs_convert and not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        if not shutil.which("LoRAConverter"):
            print("[error] LoRAConverter not found on $PATH")
            sys.exit(1)

    # Load existing JSON and backup
    entries = []
    if needs_convert:
        entries = load_json(json_path)
        if not args.dry_run and json_path.exists():
            backup_json(json_path)
            print(f"[backup] {json_path.name} → {BACKUP_DIR}/")

    mode = "full pipeline"
    if args.fix_only:
        mode = "fix-only"
    elif args.tag_only:
        mode = "tag-only"
    elif args.skip_convert:
        mode = "tag + fix (skip-convert)"

    print(f"\nProcessing {len(files)} file(s)  [{mode}]")
    if needs_convert:
        print(f"  output:  {output_dir}")
        print(f"  json:    {json_path}")

    # Process each file
    success = 0
    skipped = 0

    for f in files:
        ok = process_file(
            src_path=str(f),
            output_dir=output_dir,
            json_path=json_path,
            entries=entries,
            cli_version=cli_version,
            cli_name=args.name,
            scale_factor=scale_factor,
            non_interactive=non_interactive,
            size_limit_mb=size_limit_mb,
            dry_run=args.dry_run,
            skip_fix=args.skip_fix,
            skip_civitai=args.skip_civitai,
            skip_convert=args.skip_convert,
            fix_only=args.fix_only,
            tag_only=args.tag_only,
        )
        if ok:
            success += 1
        else:
            skipped += 1

    # Save JSON if any changes were made
    if needs_convert and not args.dry_run and success > 0:
        save_json(json_path, entries)
        print(f"\n[saved] {json_path}")

    # Summary
    print(f"\n{'═'*60}")
    print(f"  Done: {success} processed, {skipped} skipped/failed")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
