"""
Utilities for mirroring stdout/stderr to both console and log files.
"""

import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Iterator


class _TeeStream:
    """File-like object that writes to multiple streams."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        self.flush()
        return len(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()


@contextmanager
def tee_output(model_name: str, log_dir: Path = Path("logs")) -> Iterator[Path]:
    """
    Mirror stdout/stderr to console and to a timestamped log file.

    Args:
        model_name: Name of the model/pipeline, used in the log filename.
        log_dir: Directory where logs will be stored (default: ./logs).

    Yields:
        Path to the log file capturing the session.
    """

    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{model_name}_{timestamp}.txt"

    with log_path.open("w", encoding="utf-8") as log_file:
        tee_stream = _TeeStream(sys.__stdout__, log_file)
        with redirect_stdout(tee_stream), redirect_stderr(tee_stream):
            print(f"[logging] Streaming output to {log_path}")
            yield log_path

