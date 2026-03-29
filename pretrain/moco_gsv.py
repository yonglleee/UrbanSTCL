"""Wrapper entrypoint for UrbanSTCL training scripts.

`UrbanSTCL/pretrain/train.sh` expects to run `python moco_gsv.py ...` from this
folder. The actual implementation lives in `moco-v3/moco_gsv.py`.

This wrapper keeps the original scripts working without duplicating code.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    target = repo_root / "moco-v3" / "moco_gsv.py"

    if not target.is_file():
        raise FileNotFoundError(
            f"Expected training entrypoint at {target} but it was not found. "
            "Make sure you have added moco-v3/moco_gsv.py."
        )

    # Ensure imports like `import moco.*` and `import vits` resolve from moco-v3.
    sys.path.insert(0, str(target.parent))

    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
