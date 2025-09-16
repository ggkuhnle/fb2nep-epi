"""
FB2NEP bootstrap (Colab + local)

Do in a notebook (top cell):

    import runpy, pathlib
    for p in ["scripts/bootstrap.py","../scripts/bootstrap.py","../../scripts/bootstrap.py"]:
        if pathlib.Path(p).exists():
            print(f"Bootstrapping via: {p}")
            runpy.run_path(p)
            break
    else:
        raise FileNotFoundError("scripts/bootstrap.py not found")

    # Now: df, CTX, CSV_REL, REPO_ROOT, IN_COLAB are available
    df.head()

Or import style:

    from scripts.bootstrap import init
    df, ctx = init()
"""

from __future__ import annotations
import os
import sys
import shlex
import pathlib
import subprocess
from dataclasses import dataclass

# -----------------------
# Configuration (overridable via env)
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
    """Run shell command; return exit code."""
    print(">", cmd)
    return subprocess.run(shlex.split(cmd), check=False).returncode

def _looks_like_root(p: pathlib.Path) -> bool:
    return (p / "scripts").exists() and (p / "notebooks").exists()

def ensure_repo_root() -> pathlib.Path:
    """
    Ensure CWD is the repo root (contains scripts/ and notebooks/).
    If opened from notebooks/, go one level up.
    In Colab without the repo, clone it.
    """
    here = pathlib.Path.cwd()
    if _looks_like_root(here):
        return here
    if _looks_like_root(here.parent):
        os.chdir(here.parent)
        return here.parent
    if IN_COLAB:
        print("Cloning repository (Colab)…")
        if not (pathlib.Path("/content") / REPO_NAME).exists():
            _run(f"git clone {REPO_URL}")
        if (pathlib.Path.cwd() / REPO_NAME).exists():
            os.chdir(REPO_NAME)
            return pathlib.Path.cwd()
    # Fallback: stay where we are (use relative paths)
    print("⚠️ Could not auto-detect repo root; continuing in", here)
    return here

def ensure_deps():
    """
    Light dependency check. In Colab, install from requirements if needed.
    Locally, only warn (respect the user’s venv).
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
    Ensure dataset exists.
    - If file exists and not FORCE_REGEN → load it.
    - If FORCE_REGEN or missing → try to run generator.
    - In Colab, if still missing → prompt for manual upload.
    """
    if os.path.exists(csv_rel) and not FORCE_REGEN:
        print(f"Dataset found: {csv_rel} ✅")
        return

    # Try generator
    if FORCE_REGEN or not os.path.exists(csv_rel):
        if os.path.exists(gen_script):
            print("Generating dataset…")
            rc = _run(f"python {gen_script}")
            if rc == 0 and os.path.exists(csv_rel):
                print(f"Generated: {csv_rel} ✅")
                return
            print("⚠️ Generation failed or file still missing.")

    # Colab fallback: manual upload
    if IN_COLAB and not os.path.exists(csv_rel):
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

    if not os.path.exists(csv_rel):
        raise FileNotFoundError(f"Could not obtain dataset at {csv_rel}")

def init():
    """
    Initialise notebooks. Returns (df, ctx) and also binds:
    df, CTX, CSV_REL, REPO_ROOT, IN_COLAB into globals for `%run` style.
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
    globals().update({
        "df": df,
        "CTX": ctx,
        "CSV_REL": CSV_REL,
        "REPO_ROOT": repo_root,
        "IN_COLAB": IN_COLAB,
    })
    return df, ctx

if __name__ == "__main__":
    init()
