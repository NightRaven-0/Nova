#!/usr/bin/env python
"""Download a model's weights over a hostile connection, resumably.

Background (measured on this machine, everything tunnelled through a VPN):

* `ollama pull` uses a fast CDN (~1.1 MB/s) but splits the blob into 16 parts
  and only writes a part's progress to disk when that whole ~580 MB part
  finishes. One dropped connection and it restarts from zero: 11 attempts
  produced 0 MB of durable progress.
* Hugging Face resumes properly but routes through its Xet backend, which only
  managed 76-250 KB/s here, and parallel range requests mostly time out.

So: take Ollama's fast CDN and add our own byte-level resume. This downloads the
weights blob directly, flushing every few seconds, verifies its sha256, and
drops it into Ollama's blob store under the name Ollama expects. A subsequent
`ollama pull` then finds the big blob already present and only fetches the few
KB of template/params/manifest.

    python scripts/fetch_model.py ollama qwen3:14b
    python scripts/fetch_model.py ollama qwen3.6:35b-a3b --streams 8
    python scripts/fetch_model.py hf Qwen/Qwen3-14B-GGUF Qwen3-14B-Q4_K_M.gguf

Resume by rerunning the same command. Progress is also written to
<file>.status.json so it can be read without parsing the log. Uses urllib, which
honours the Windows system proxy (curl does not).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
import urllib.request

CHUNK = 512 * 1024
READ_TIMEOUT = 60   # a stalled read raises instead of hanging; that stream resumes
FLUSH_EVERY = 5.0   # seconds between status writes
REGISTRY = "https://registry.ollama.ai/v2/library"
MANIFEST_ACCEPT = "application/vnd.docker.distribution.manifest.v2+json"


def human(n: float) -> str:
    return "{:.2f} GB".format(n / 1e9) if n >= 1e9 else "{:.0f} MB".format(n / 1e6)


def get_json(url: str, accept: str = "", tries: int = 20):
    """Small JSON fetch, retried: a blip here must not kill a multi-hour job."""
    headers = {"User-Agent": "nova-fetch"}
    if accept:
        headers["Accept"] = accept
    for attempt in range(1, tries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            return json.load(urllib.request.urlopen(req, timeout=60))
        except Exception as e:
            if attempt == tries:
                raise
            print("  metadata fetch failed ({}); retry {}/{}".format(
                str(e)[:60], attempt, tries), flush=True)
            time.sleep(min(3 * attempt, 30))


# ---------------------------------------------------------------------------
# the download engine
# ---------------------------------------------------------------------------
class Part:
    __slots__ = ("start", "end", "done", "attempts", "error")

    def __init__(self, start, end, done=0):
        self.start, self.end, self.done = start, end, done
        self.attempts, self.error = 0, ""

    @property
    def size(self):
        return self.end - self.start + 1

    @property
    def complete(self):
        return self.done >= self.size


def _load_parts(status_path, total, streams):
    try:
        with open(status_path, encoding="utf-8") as f:
            saved = json.load(f)
        if saved.get("total") == total and len(saved.get("parts", [])) == streams:
            parts = [Part(p["start"], p["end"], p["done"]) for p in saved["parts"]]
            print("  resuming at {}".format(human(sum(p.done for p in parts))), flush=True)
            return parts
    except Exception:
        pass
    step = total // streams
    return [Part(i * step, total - 1 if i == streams - 1 else (i + 1) * step - 1)
            for i in range(streams)]


def _save_status(status_path, parts, total, done_flag=False):
    payload = {"total": total, "done": done_flag,
               "have": sum(p.done for p in parts), "ts": time.time(),
               "parts": [{"start": p.start, "end": p.end, "done": p.done,
                          "attempts": p.attempts, "error": p.error} for p in parts]}
    try:
        tmp = status_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, status_path)
    except OSError:
        pass


def _worker(url, path, part, stop):
    """Pull one byte range, flushing as it goes; retry forever on any failure."""
    while not stop.is_set() and not part.complete:
        part.attempts += 1
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "nova-fetch",
                "Range": "bytes={}-{}".format(part.start + part.done, part.end),
            })
            with urllib.request.urlopen(req, timeout=READ_TIMEOUT) as r:
                with open(path, "r+b") as f:
                    f.seek(part.start + part.done)
                    while not stop.is_set():
                        buf = r.read(CHUNK)
                        if not buf:
                            break
                        f.write(buf)
                        part.done += len(buf)
                        if part.complete:
                            break
            part.error = ""
        except Exception as e:
            part.error = str(e)[:90]
            time.sleep(min(3 + part.attempts, 30))


def download(url: str, out_path: str, total: int, sha256: str, streams: int) -> bool:
    """Fetch `url` to `out_path`, resumably. Verifies sha256 before committing."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    part_path = out_path + ".part"
    status_path = out_path + ".status.json"

    if os.path.exists(out_path) and os.path.getsize(out_path) == total:
        print("already complete: " + out_path, flush=True)
        return True

    print("  expecting {} (sha256 {}) over {} stream(s)".format(
        human(total), (sha256 or "unknown")[:16], streams), flush=True)
    parts = _load_parts(status_path, total, streams)
    with open(part_path, "r+b" if os.path.exists(part_path) else "wb") as f:
        f.truncate(total)  # preallocate so each stream can seek to its own offset

    stop = threading.Event()
    threads = [threading.Thread(target=_worker, args=(url, part_path, p, stop),
                                name="part{}".format(i), daemon=True)
               for i, p in enumerate(parts) if not p.complete]
    for t in threads:
        t.start()

    started, last = time.time(), sum(p.done for p in parts)
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(FLUSH_EVERY)
            have = sum(p.done for p in parts)
            rate = (have - last) / FLUSH_EVERY
            last = have
            _save_status(status_path, parts, total)
            eta = (total - have) / rate / 3600 if rate > 1000 else 0
            print("  {:>9s} / {} ({:4.1f}%)  {:5.0f} KB/s  {}  {}/{} streams live".format(
                human(have), human(total), 100.0 * have / total, rate / 1024,
                "ETA {:4.1f} h".format(eta) if eta else "stalled",
                sum(1 for p in parts if not p.complete), len(parts)), flush=True)
    except KeyboardInterrupt:
        stop.set()
        _save_status(status_path, parts, total)
        print("stopped; rerun the same command to resume", flush=True)
        return False

    _save_status(status_path, parts, total)
    have = sum(p.done for p in parts)
    if have != total:
        print("incomplete: {} of {}".format(human(have), human(total)), flush=True)
        return False

    if sha256:
        print("  verifying sha256 of {}...".format(human(total)), flush=True)
        h = hashlib.sha256()
        with open(part_path, "rb") as f:
            for block in iter(lambda: f.read(8 * CHUNK), b""):
                h.update(block)
        if h.hexdigest() != sha256:
            bad = part_path + ".corrupt"
            os.replace(part_path, bad)
            os.path.exists(status_path) and os.remove(status_path)
            print("  SHA MISMATCH -- kept as {}; delete it and rerun".format(bad), flush=True)
            return False
        print("  sha256 ok", flush=True)

    os.replace(part_path, out_path)
    _save_status(status_path, parts, total, done_flag=True)
    print("complete: {} in {:.1f} h".format(out_path, (time.time() - started) / 3600), flush=True)
    return True


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------
def from_ollama(tag: str, streams: int, models_dir: str) -> bool:
    """Fetch a model's weights blob from Ollama's own CDN into its blob store."""
    name, _, version = tag.partition(":")
    version = version or "latest"
    manifest = get_json("{}/{}/manifests/{}".format(REGISTRY, name, version), MANIFEST_ACCEPT)
    layers = sorted(manifest.get("layers", []), key=lambda l: l.get("size", 0), reverse=True)
    if not layers:
        print("no layers in the manifest for " + tag)
        return False

    # Grab every layer big enough to hurt if it had to restart. Some models have
    # more than one: qwen3.6:35b-a3b ships a 21.7 GB weights layer AND a 902 MB
    # second layer, and skipping the smaller one left `ollama pull` to fetch it
    # on the fragile path.
    big = [l for l in layers if l.get("size", 0) >= 50 * 1024 * 1024]
    small = [l for l in layers if l not in big]
    blobs = os.path.join(models_dir, "blobs")
    print("{}: {} layer(s) >= 50 MB to fetch here, {} small layer(s) ({}) left to "
          "`ollama pull`".format(tag, len(big), len(small),
                                 human(sum(l.get("size", 0) for l in small))), flush=True)

    for i, layer in enumerate(big, 1):
        digest = layer["digest"].split(":")[-1]
        out = os.path.join(blobs, "sha256-" + digest)
        print("[{}/{}] {} -> {}".format(i, len(big), human(layer["size"]), out), flush=True)
        url = "{}/{}/blobs/sha256:{}".format(REGISTRY, name, digest)
        if not download(url, out, layer["size"], digest, streams):
            return False
    print("\nnow run:  ollama pull {}".format(tag), flush=True)
    print("(it will see these blobs and only fetch the small remaining layers)", flush=True)
    return True


def from_hf(repo: str, filename: str, streams: int, dest: str) -> bool:
    """Fetch a .gguf from Hugging Face (slower here, but keeps provenance)."""
    files = get_json("https://huggingface.co/api/models/{}/tree/main".format(repo))
    for entry in files:
        if entry.get("path") == filename:
            lfs = entry.get("lfs") or {}
            total = lfs.get("size") or entry.get("size") or 0
            out = os.path.join(dest, os.path.basename(filename))
            print("{} -> {}".format(filename, out), flush=True)
            url = "https://huggingface.co/{}/resolve/main/{}?download=true".format(repo, filename)
            return download(url, out, total, lfs.get("oid"), streams)
    print("{} not found in {}".format(filename, repo))
    return False


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="source", required=True)

    o = sub.add_parser("ollama", help="from Ollama's registry/CDN (fastest here)")
    o.add_argument("tag", help='e.g. "qwen3:14b"')
    o.add_argument("--streams", type=int, default=8)
    o.add_argument("--models-dir", default=os.environ.get("OLLAMA_MODELS", r"Z:\ollama\models"))

    h = sub.add_parser("hf", help="from Hugging Face (slower here)")
    h.add_argument("repo")
    h.add_argument("filename")
    h.add_argument("--streams", type=int, default=1)
    h.add_argument("--dest", default=r"Z:\ollama\gguf")

    a = p.parse_args()
    if a.source == "ollama":
        ok = from_ollama(a.tag, max(1, a.streams), a.models_dir)
    else:
        ok = from_hf(a.repo, a.filename, max(1, a.streams), a.dest)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
