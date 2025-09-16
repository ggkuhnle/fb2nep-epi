"""
FB2NEP notebook bootstrap (Colab + local)

Usage (at the very top of each notebook):

    %run ../notebooks/_bootstrap.py
    # now: df, CSV_REL, REPO_ROOT, IN_COLAB are defined

Or:

    from notebooks._bootstrap import init
    df, ctx = init()

Default: load the committed CSV if present.
Regeneration only happens if FB2NEP_FORCE_REGEN=1 is set.
"""

from __future__ import annotations
import os
import sys
import shlex
import pathlib
import subprocess
from dataclasses import dataclass

# -----------------------
# Configuration
# -----------------------
REPO_NAME   = os.getenv("FB2NEP_REPO", "fb2nep-epi")
REPO_URL    = os.getenv("FB2NEP_REPO_URL", "https://github.com/ggkuhnle/fb2nep-epi.git")
CSV_REL     = os.getenv("FB2NEP_CSV", "data/synthetic/fb2nep.csv")
GEN_SCRIPT  = os.getenv("FB2NEP_GEN", "scripts/generate_dataset.py")
REQS_FILE   = os.getenv("FB2NEP_REQS", "requirements.txt")
FORCE_REGEN = os.getenv("FB2NEP_FORCE_REGEN", "0") == "1"

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
    """Ensure CWD is the repo root (scripts/ + notebooks/ present)."""
    here = pathlib.Path.cwd()
    def looks_like_root(p: pathlib.Path) -> bool:
        return (p / "scripts").exists() and (p / "notebooks").exists()
    if looks_like_root(here): return here
    if looks_like_root(here.parent):
        os.chdir(here.parent)
        return here.parent
    if IN_COLAB:
        print("Cloning repository (Colab)…")
        if not (pathlib.Path("/content") / REPO_NAME).exists():
            _run(f"git clone {REPO_URL}")
        if (pathlib.Path.cwd() / REPO_NAME).exists():
            os.chdir(REPO_NAME)
            return pathlib.Path.cwd()
    return here

def ensure_deps():
    """Light dependency check; install in Colab if missing."""
    try:
        import numpy, pandas, matplotlib, statsmodels  # noqa: F401
    except Exception as e:
        if IN_COLAB:
            print("Installing dependencies in Colab…")
            if os.path.exists(REQS_FILE):
                _run(f"pip install -q -r {REQS_FILE}")
            else:
                _run("pip install -q numpy pandas matplotlib seaborn statsmodels")
        else:
            print("⚠️ Missing deps locally:", e)

def ensure_data(csv_rel: str, gen_script: str):
    """
    Ensure dataset exists.
    - If file exists → load it.
    - If FORCE_REGEN=1 → run generator.
    - Else (missing + no regen) → Colab upload fallback.
    """
    if os.path.exists(csv_rel) and not FORCE_REGEN:
        print(f"Dataset found: {csv_rel} ✅")
        return

    if FORCE_REGEN or not os.path.exists(csv_rel):
        if os.path.exists(gen_script):
            print("Generating dataset…")
            rc = _run(f"python {gen_script}")
            if rc == 0 and os.path.exists(csv_rel):
                print(f"Generated: {csv_rel} ✅")
                return
            print("⚠️ Generation failed.")

    if IN_COLAB and not os.path.exists(csv_rel):
        try:
            from google.colab import files  # type: ignore
            os.makedirs(os.path.dirname(csv_rel), exist_ok=True)
            print(f"Upload fb2nep.csv (will be saved to {csv_rel}) …")
            uploaded = files.upload()
            if "fb2nep.csv" in uploaded:
                with open(csv_rel, "wb") as f:
                    f.write(uploaded["fb2nep.csv"])
                print(f"Uploaded: {csv_rel} ✅")
                return
        except Exception as e:
            print("Upload fallback failed:", e)

    if not os.path.exists(csv_rel):
        raise FileNotFoundError(f"Could not obtain dataset at {csv_rel}")

def init():
    """Return (df, ctx). Also binds df, CSV_REL, REPO_ROOT, IN_COLAB for `%run` use."""
    import pandas as pd
    repo_root = ensure_repo_root()
    ensure_deps()
    ensure_data(CSV_REL, GEN_SCRIPT)
    df = pd.read_csv(CSV_REL)
    print(df.shape, "— dataset ready")
    ctx = Context(repo_root, CSV_REL, GEN_SCRIPT, IN_COLAB, REPO_URL, REPO_NAME)
    globals().update({"df": df, "CSV_REL": CSV_REL, "REPO_ROOT": repo_root, "IN_COLAB": IN_COLAB, "CTX": ctx})
    return df, ctx

if __name__ == "__main__":
    init()
