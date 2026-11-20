"""Reading the single-PSF benchmark results (shared by bench.py and report.py)."""
import json
from pathlib import Path

RENAMED = {'box': 'midpoint', 'spectral': 'exact'}   # pixel-integration names stored by runs before the rename


def load_results(path):
    """A results JSON, with the pre-rename pixel-integration names ('box', 'spectral') mapped to the current ones."""
    def fix(x):
        if isinstance(x, dict):
            return {k: RENAMED.get(v, v) if k == 'name' and isinstance(v, str) else fix(v) for k, v in x.items()}
        return [fix(v) for v in x] if isinstance(x, list) else x
    return fix(json.loads(Path(path).read_text()))
