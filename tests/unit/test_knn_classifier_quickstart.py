from pathlib import Path

import numpy as np


def test_readme_quickstart_snippet_runs():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "## Quickstart" in readme, "README missing Quickstart section"

    quickstart_block = readme.split("## Quickstart", maxsplit=1)[1]
    code_start = quickstart_block.find("```python")
    assert code_start != -1, "Quickstart Python block not found"

    code_start += len("```python\n")
    code_end = quickstart_block.find("```", code_start)
    assert code_end != -1, "Quickstart Python block not terminated"

    code = quickstart_block[code_start:code_end].strip()
    namespace = {}
    exec(code, namespace)
    predictions = namespace.get("predictions")
    assert predictions is not None, "Snippet should define `predictions`"
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (1,)
    assert predictions[0] in namespace["y"]