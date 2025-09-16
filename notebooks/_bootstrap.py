"""
FB2NEP notebook bootstrap (Colab + local)

Usage (at the very top of each notebook):

    %run ../notebooks/_bootstrap.py
    # now: df, CSV_REL, REPO_ROOT, IN_COLAB are defined

Or (import style):

    from notebooks._bootstrap import init
    df, ctx = init()  # ctx has paths and flags

British English; minimal side effects; transparent data generation.
"""

from __future__ import annotations
import os
import sys
import shlex
import pathlib
import subprocess
from dataclasses import dataclass

# -----------------------
# Configuration (override via env vars if needed)
# -----------------------
REPO_NAME   = os.getenv("FB2NEP_REPO", "fb2nep-epi")
REPO_URL    = os.getenv("FB2NEP_REPO_URL", "https://github.com/ggkuhnle/fb2nep-epi.git")
CSV_REL     = os.getenv("FB2NEP_CSV", "data/synthetic/fb2nep.csv")
GEN_SCRIPT  = os.getenv("FB2NEP_GEN", "scripts/generate_dataset.py")
REQS_FILE   = os.getenv("FB2NEP_REQS", "requirements.txt")

IN_COLAB = "google.colab" in sys.modules

@dataclass
class Context:
    repo_root: pathlib.Path
    csv_rel: str
    gen_script: str
    in_colab: bool
    repo_url: str
    repo_name: str

def _run(cmd: str) -> int:
    """Run a shell command verbosely; return exit code."""
    print(">", cmd)
    return subprocess.run(shlex.split(cmd), check=False).returncode

def ensure_repo_root() -> pathlib.Path:
    """
    Ensure CWD is the repo root (contains scripts/ and notebooks/).
    If opened from notebooks/, go one level up.
    If in Colab without the repo, clone it.
    """
    here = pathlib.Path.cwd()
    def looks_like_root(p: pathlib.Path) -> bool:
        return (p / "scripts").exists() and (p / "notebooks").exists()

    if looks_like_root(here):
        return here
    if looks_like_root(here.parent):
        os.chdir(here.parent)
        return here.parent
    if IN_COLAB:
        print("Cloning repository (Colab)…")
        if not (pathlib.Path("/content") / REPO_NAME).exists():
            rc = _run(f"git clone {REPO_URL}")
            if rc != 0:
                print("⚠️ git clone failed; will continue (you may need manual upload later).")
        # cd into repo if present
        if (pathlib.Path.cwd() / REPO_NAME).exists():
            os.chdir(REPO_NAME)
            return pathlib.Path.cwd()
        # last resort: stay where we are
        return pathlib.Path.cwd()
    # Local: assume user launched from elsewhere; do not chdir blindly
    print("⚠️ Could not auto-detect repo root; continuing in", here)
    return here

def ensure_deps():
    """
    Light dependency check. If core libs missing in Colab,
    install from requirements.txt. Locally, just warn.
    """
    try:
        import numpy, pandas, matplotlib, statsmodels  # noqa: F401
    except Exception as e:
        if IN_COLAB:
            print("Installing Python dependencies (Colab)…")
            if os.path.exists(REQS_FILE):
                _run(f"pip install -q -r {REQS_FILE}")
            else:
                _run("pip install -q numpy pandas matplotlib seaborn statsmodels")
        else:
            print("⚠️ Missing dependencies locally:", e)
            print("   Consider: `pip install -r requirements.txt` in your virtual environment.")

def ensure_data(csv_rel: str, gen_script: str):
    """
    Ensure dataset CSV exists. Prefer generating via the script for transparency.
    Colab fallback: manual file upload.
    """
    if os.path.exists(csv_rel):
        print(f"Dataset found: {csv_rel} ✅")
        return

    # Try to generate via the script
    if os.path.exists(gen_script):
        print("Generating dataset…")
        rc = _run(f"python {gen_script}")
        if rc == 0 and os.path.exists(csv_rel):
            print(f"Generated: {csv_rel} ✅")
            return
        print("⚠️ Generation failed or file still missing.")

    # Colab fallback: manual upload
    if IN_COLAB:
        try:
            from google.colab import files  # type: ignore
            target_dir = os.path.dirname(csv_rel) or "."
            os.makedirs(target_dir, exist_ok=True)
            print(f"Upload fb2nep.csv (will be saved to {csv_rel}) …")
            uploaded = files.upload()
            if "fb2nep.csv" in uploaded:
                with open(csv_rel, "wb") as f:
                    f.write(uploaded["fb2nep.csv"])
                print(f"Uploaded: {csv_rel} ✅")
                return
            else:
                print("⚠️ fb2nep.csv not provided.")
        except Exception as e:
            print("Upload fallback failed:", e)

    raise FileNotFoundError(f"Could not obtain dataset at {csv_rel}")

def init():
    """
    One-call initialiser for notebooks.
    Returns (df, ctx) where ctx is a Context dataclass with useful info.
    Also binds df, CSV_REL, REPO_ROOT, IN_COLAB into globals for %run use.
    """
    import pandas as pd

    repo_root = ensure_repo_root()
    ensure_deps()
    ensure_data(CSV_REL, GEN_SCRIPT)

    df = pd.read_csv(CSV_REL)
    print(df.shape, "— dataset ready")

    ctx = Context(
        repo_root=repo_root,
        csv_rel=CSV_REL,
        gen_script=GEN_SCRIPT,
        in_colab=IN_COLAB,
        repo_url=REPO_URL,
        repo_name=REPO_NAME,
    )

    # Expose a few conveniences to `%run` users
    globals().update({
        "df": df,
        "CSV_REL": CSV_REL,
        "REPO_ROOT": repo_root,
        "IN_COLAB": IN_COLAB,
        "CTX": ctx,
    })
    return df, ctx

# If executed via `%run notebooks/_bootstrap.py`, call init and leave df in global scope.
if __name__ == "__main__":
    init()
