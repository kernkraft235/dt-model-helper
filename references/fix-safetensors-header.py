#!/usr/bin/env python3
"""
Fix malformed safetensors headers for Draw Things / MFLUX compatibility.
- Strips __metadata__ block
- Strips "model.diffusion_model." prefix from tensor keys

Accepts one or more .safetensors file paths as arguments (Automator folder
actions pass all dropped files as $@ to a shell script, or as a list to a
Run Shell Script action — wire stdin lines or $1..$n depending on how you
call this).

Overwrite behavior:
  - Non-interactive (Automator, launchd, pipe): always overwrites in-place
  - Interactive shell: requires --overwrite flag; otherwise writes -FIXED copy

xattrs written:
  kernkraft235.safetensors.pre-sha256sum
      SHA-256 hex digest of the file before any modification.
      Written before any fix is applied. Unrecoverable if fix-v1 > 0 already
      exists without it.

  kernkraft235.safetensors.fix-v1
      Bitmask (decimal string) of header fixes applied:
        0 = no changes needed
        1 = __metadata__ stripped
        2 = model.diffusion_model. prefix stripped from keys
        3 = both

  kernkraft235.civitai.<field>
      Flattened CivitAI model-version metadata fetched by sha256 hash.
      Fields written (when present in API response):
        id, modelId, nsfwLevel, baseModel, triggerWords, name, modelType,
        air, fileName, description, images
      triggerWords and images are stored as JSON arrays.
      modelId is also used to write com.apple.metadata:kMDItemWhereFroms
      as a binary plist array containing the canonical CivitAI model URL.
"""

import sys
import struct
import json
import os
import argparse
import hashlib
import plistlib
import urllib.request
import urllib.error
from pathlib import Path

# ── xattr helpers (ctypes against macOS libc, no third-party deps) ────────────

import ctypes
import ctypes.util

_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)

XATTR_FIX    = b"kernkraft235.safetensors.fix-v1"
XATTR_SHA256 = b"kernkraft235.safetensors.pre-sha256sum"
XATTR_WHERE  = b"com.apple.metadata:kMDItemWhereFroms"
CIVITAI_PFX  = "kernkraft235.civitai."

# Fields to extract and their source paths in the API response.
# Value is either a top-level key, a dotted path, or a callable.
CIVITAI_FIELDS = {
    "id":           lambda d: d.get("id"),
    "modelId":      lambda d: d.get("modelId"),
    "nsfwLevel":    lambda d: d.get("nsfwLevel"),
    "baseModel":    lambda d: d.get("baseModel"),
    "triggerWords": lambda d: d.get("trainedWords"),      # renamed
    "name":         lambda d: d.get("model", {}).get("name"),
    "modelType":    lambda d: d.get("model", {}).get("type"),
    "air":          lambda d: d.get("air"),
    "fileName":     lambda d: (d.get("files") or [{}])[0].get("name"),
    "description":  lambda d: d.get("description"),
    "images":       lambda d: [i["url"] for i in d.get("images", []) if "url" in i] or None,
}


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

def get_fix_xattr(path: Path) -> bytes | None:
    return _xattr_get(str(path), XATTR_FIX)

def get_sha256_xattr(path: Path) -> bytes | None:
    return _xattr_get(str(path), XATTR_SHA256)

def civitai_xattr_exists(path: Path) -> bool:
    """True if any kernkraft235.civitai.* key is already set."""
    key = (CIVITAI_PFX + "id").encode()
    return _xattr_get(str(path), key) is not None

def mark_fix(path: Path, fix_flags: int) -> None:
    _xattr_set(str(path), XATTR_FIX, str(fix_flags).encode())

def mark_sha256(path: Path, hexdigest: str) -> None:
    _xattr_set(str(path), XATTR_SHA256, hexdigest.encode())

def mark_civitai(path: Path, data: dict) -> int:
    """Write extracted CivitAI fields as individual xattrs. Returns count written."""
    written = 0
    for field, extractor in CIVITAI_FIELDS.items():
        value = extractor(data)
        if value is None:
            continue
        # Arrays stored as compact JSON, scalars as plain strings
        if isinstance(value, (list, dict)):
            raw = json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()
        else:
            raw = str(value).encode()
        _xattr_set(str(path), (CIVITAI_PFX + field).encode(), raw)
        written += 1
    return written

def mark_where_froms(path: Path, url: str) -> None:
    """Overwrite kMDItemWhereFroms with a single-element binary plist array."""
    plist_bytes = plistlib.dumps([url], fmt=plistlib.FMT_BINARY)
    _xattr_set(str(path), XATTR_WHERE, plist_bytes)


# ── sha256 ────────────────────────────────────────────────────────────────────

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(16 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ── CivitAI API ───────────────────────────────────────────────────────────────

CIVITAI_BY_HASH = "https://civitai.com/api/v1/model-versions/by-hash/{}"
CIVITAI_MODEL   = "https://civitai.com/models/{}"

def fetch_civitai(sha256: str) -> dict | None:
    url = CIVITAI_BY_HASH.format(sha256.upper())
    req = urllib.request.Request(url, headers={"User-Agent": "fix-safetensors-header/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    except Exception:
        return None


# ── safetensors header logic ──────────────────────────────────────────────────

PREFIX = "model.diffusion_model."

FIX_METADATA = 1
FIX_KEYS     = 2

def read_header(fin) -> dict:
    header_len = struct.unpack("<Q", fin.read(8))[0]
    return json.loads(fin.read(header_len).decode("utf-8"))

def detect_fixes(header: dict) -> int:
    flags = 0
    if "__metadata__" in header:
        flags |= FIX_METADATA
    if any(k.startswith(PREFIX) for k in header):
        flags |= FIX_KEYS
    return flags

def build_fixed_header(header: dict, flags: int) -> bytes:
    if flags & FIX_METADATA:
        header.pop("__metadata__", None)
    new_header = {}
    for k, v in header.items():
        if (flags & FIX_KEYS) and k.startswith(PREFIX):
            new_header[k[len(PREFIX):]] = v
        else:
            new_header[k] = v
    return json.dumps(new_header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


# ── per-file orchestration ────────────────────────────────────────────────────

def process_file(src: Path, overwrite: bool) -> None:
    if src.suffix != ".safetensors":
        print(f"[skip]   {src.name}  (not a .safetensors file)")
        return

    fix_xattr = get_fix_xattr(src)
    sha_xattr = get_sha256_xattr(src)
    fix_done  = fix_xattr is not None
    fix_flags = int(fix_xattr) if fix_done else None
    sha_done  = sha_xattr is not None

    # Already fixed but sha256 was never captured — unrecoverable for CivitAI lookup
    if fix_done and fix_flags > 0 and not sha_done:
        print(f"[warn]   {src.name}  (already fixed, pre-modification sha256 unrecoverable)")
        return

    # Compute sha256 before touching anything if we don't have it yet
    if not sha_done:
        digest = sha256_file(src)
        mark_sha256(src, digest)
        print(f"[sha256] {src.name}  {digest}")
    else:
        digest = sha_xattr.decode()

    # ── header fix ───────────────────────────────────────────────────────────

    if fix_done and fix_flags == 0:
        pass  # already confirmed clean, fall through to CivitAI
    elif fix_done:
        pass  # already fixed, fall through to CivitAI
    else:
        with src.open("rb") as fin:
            header = read_header(fin)
            flags  = detect_fixes(header)

            if not flags:
                mark_fix(src, 0)
                print(f"[ok]     {src.name}  (no fix needed)")
                dst = src
            else:
                fixed_bytes = build_fixed_header(header, flags)

                if overwrite:
                    tmp = src.with_suffix(".tmp.safetensors")
                    dst = src
                else:
                    tmp = src.with_name(src.stem + "-FIXED.safetensors")
                    dst = tmp

                with tmp.open("wb") as fout:
                    fout.write(struct.pack("<Q", len(fixed_bytes)))
                    fout.write(fixed_bytes)
                    while True:
                        chunk = fin.read(16 * 1024 * 1024)
                        if not chunk:
                            break
                        fout.write(chunk)

                if overwrite:
                    tmp.replace(dst)
                    mark_sha256(dst, digest)  # re-apply — overwrite lost the xattr
                else:
                    mark_sha256(dst, digest)

                fix_desc = "+".join(
                    n for bit, n in ((FIX_METADATA, "metadata"), (FIX_KEYS, "keys"))
                    if flags & bit
                )
                mark_fix(dst, flags)
                print(f"[fixed]  {dst.name}  (xattr={flags}: {fix_desc})")
                src = dst  # subsequent xattrs go on the live file

    # ── CivitAI metadata ──────────────────────────────────────────────────────

    if civitai_xattr_exists(src):
        print(f"[skip]   {src.name}  (CivitAI xattrs already present)")
        return

    civitai = fetch_civitai(digest)
    if civitai is None:
        print(f"[civitai] {src.name}  (not found or API error)")
        return

    n = mark_civitai(src, civitai)

    model_id = civitai.get("modelId")
    if model_id is not None:
        model_url = CIVITAI_MODEL.format(model_id)
        mark_where_froms(src, model_url)
        print(f"[civitai] {src.name}  {n} fields written, WhereFroms → {model_url}")
    else:
        print(f"[civitai] {src.name}  {n} fields written (no modelId for WhereFroms)")


# ── entry point ───────────────────────────────────────────────────────────────

def is_interactive() -> bool:
    return (
        sys.stdin.isatty()
        and sys.stdout.isatty()
        and os.environ.get("TERM") not in (None, "dumb")
    )

def main():
    parser = argparse.ArgumentParser(
        description="Fix safetensors headers and tag with CivitAI metadata."
    )
    parser.add_argument("inputs", nargs="+", help="One or more .safetensors files")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite input file(s) in-place (always active when non-interactive)",
    )
    args = parser.parse_args()

    overwrite = args.overwrite or not is_interactive()

    for path_str in args.inputs:
        src = Path(path_str)
        if not src.exists():
            print(f"[error]  {path_str}  (file not found)")
            continue
        if not src.is_file():
            print(f"[error]  {path_str}  (not a regular file)")
            continue
        process_file(src, overwrite)

if __name__ == "__main__":
    main()
