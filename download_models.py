"""Pre-download both model checkpoints so the first WebSocket isn't a silent 3GB pull.

The Hugging Face Xet transfer backend stalls on some networks: bytes stop arriving
and nothing ever times out. So the parent process watches cache growth and, if the
download wedges, restarts the child with Xet disabled (plain HTTP, slower but steady).

Note: huggingface_hub picks a fresh temp name per download session, so a restart does
NOT resume — it starts the file over and strands the previous partial. That makes the
stall watchdog expensive, hence the deliberately generous STALL_SECONDS and the
cleanup of orphaned .incomplete files before each retry.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import vlm

# Generous: a restart throws away all progress, so only fire on a genuine wedge,
# never on a merely slow link.
STALL_SECONDS = 180
POLL_SECONDS = 10


def hub_dir() -> Path:
    root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return root / "hub"


def cache_bytes() -> int:
    hub = hub_dir()
    if not hub.exists():
        return 0
    return sum(f.stat().st_size for f in hub.rglob("*") if f.is_file())


def clear_partials() -> None:
    """Drop orphaned .incomplete blobs — a restart can't resume them anyway."""
    hub = hub_dir()
    if not hub.exists():
        return
    for stale in hub.glob("models--mlx-community--*/blobs/*.incomplete"):
        try:
            stale.unlink()
        except OSError:
            pass


def run_child(disable_xet: bool) -> int:
    env = dict(os.environ)
    env["HF_HUB_DOWNLOAD_TIMEOUT"] = env.get("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    if disable_xet:
        env["HF_HUB_DISABLE_XET"] = "1"

    proc = subprocess.Popen([sys.executable, __file__, "--child"], env=env)

    last_size, last_change = cache_bytes(), time.monotonic()
    while proc.poll() is None:
        time.sleep(POLL_SECONDS)
        size = cache_bytes()
        if size != last_size:
            last_size, last_change = size, time.monotonic()
        elif time.monotonic() - last_change > STALL_SECONDS:
            print(f"\n      Download stalled for {STALL_SECONDS}s — restarting transfer.")
            proc.kill()
            proc.wait()
            return 99
    return proc.returncode


def download_vlm() -> bool:
    from huggingface_hub import snapshot_download

    print(f"[1/2] Downloading VLM weights: {vlm.MODEL_ID}")
    print("      ~3.1 GB on first run. Progress bars below; later runs are instant.")
    try:
        path = snapshot_download(vlm.MODEL_ID, revision=vlm.MODEL_REVISION)
    except Exception as exc:
        print(f"      FAILED: {exc}")
        return False
    print(f"      OK -> {path}\n")
    return True


def download_detector() -> bool:
    print("[2/2] Downloading RF-DETR Nano weights (~350 MB)")
    # Importing torch+torchvision takes 20-40s and prints nothing. Say so, or this
    # looks wedged and invites a Ctrl-C.
    print("      Importing PyTorch first (20-40s, no output — this is normal)…",
          flush=True)
    try:
        from rfdetr import RFDETRNano

        print("      PyTorch loaded. Fetching checkpoint…", flush=True)
        RFDETRNano()
    except Exception as exc:
        print(f"      FAILED: {exc}")
        return False
    print("      OK\n")
    return True


def child_main() -> int:
    ok = download_vlm()
    ok = download_detector() and ok
    return 0 if ok else 1


MAX_ATTEMPTS = 4


def parent_main() -> int:
    """Retry around stalls and transient DNS/connection drops.

    Attempt 1 uses the fast Xet backend; every retry falls back to plain HTTP,
    which is slower but far less prone to wedging. Each retry restarts the file
    from zero, so stale partials are cleared first to keep the cache from growing.
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        disable_xet = attempt > 1
        if attempt > 1:
            print(f"      Retry {attempt - 1}/{MAX_ATTEMPTS - 1} over plain HTTP "
                  f"(restarts the transfer from the beginning)…")
            clear_partials()
            time.sleep(5)

        code = run_child(disable_xet)
        if code == 0:
            print("All model weights cached.")
            return 0

    print("Model download failed after "
          f"{MAX_ATTEMPTS} attempts. See README Troubleshooting #4.")
    return 1


if __name__ == "__main__":
    sys.exit(child_main() if "--child" in sys.argv else parent_main())
