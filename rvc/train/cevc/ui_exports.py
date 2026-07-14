"""Safe Gradio exports for CEVC artifacts stored on Google Drive.

Gradio only serves files from the application working directory, the system
temporary directory, or explicitly allowed paths. CEVC experiment folders may
resolve through ``logs/<name>`` symlinks to Google Drive, so persistent files
must be copied to a temporary UI-only directory before they are returned by a
Gradio event. The originals remain untouched on Drive.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Iterable


DEFAULT_UI_CACHE = Path(tempfile.gettempdir()) / "applio_cevc2b_ui"


def publish_files_for_ui(
    paths: Iterable[str | Path],
    *,
    prefix: str,
    cache_root: str | Path | None = None,
) -> list[str]:
    """Copy existing artifacts to a fresh temporary directory for Gradio.

    Returns absolute paths below the system temporary directory. Persistent
    experiment files are never moved or deleted.
    """

    sources = [Path(path).expanduser().resolve() for path in paths]
    if not sources:
        raise ValueError("No CEVC artifacts were provided for UI export")
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(f"CEVC UI artifact does not exist: {source}")

    root = Path(cache_root).expanduser().resolve() if cache_root else DEFAULT_UI_CACHE
    root.mkdir(parents=True, exist_ok=True)
    export_dir = Path(tempfile.mkdtemp(prefix=f"{prefix}_", dir=str(root)))

    exported: list[str] = []
    used_names: set[str] = set()
    for index, source in enumerate(sources):
        name = source.name
        if name in used_names:
            name = f"{index:02d}_{name}"
        used_names.add(name)
        destination = export_dir / name
        shutil.copyfile(source, destination)
        if destination.stat().st_size != source.stat().st_size:
            raise IOError(f"Incomplete CEVC UI artifact copy: {source}")
        exported.append(str(destination))
    return exported
