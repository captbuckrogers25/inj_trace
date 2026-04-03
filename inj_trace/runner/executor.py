"""
PTMExecutor: launch the SHIELDS-PTM Fortran executable.

PTM reads all input files relative to the current working directory:
  ptm_input/ptm_parameters_XXXX.txt
  ptm_data/ptm_fields_XXXX.dat
  ptm_data/tgrid.dat

and writes output to:
  ptm_output/ptm_XXXX.dat

Therefore this executor MUST chdir into run_dir before invoking the binary.
ptm_tools.cd() is used as a context manager to restore the original directory
safely on success or failure.

Usage
-----
    exe = PTMExecutor(run_dir='./run001')
    result = exe.run_single(run_id=1)
    print(result.returncode)

    # Multiple runs in parallel
    results = exe.run_parallel([1, 2, 3, 4], max_workers=4)
"""

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional

from inj_trace.config import load_config


class PTMExecutor:
    """Launch the PTM Fortran executable for one or more run IDs.

    Parameters
    ----------
    run_dir        : directory containing ptm_input/, ptm_data/, ptm_output/
                     and the ptm executable (or accessible via ptm_executable).
    ptm_executable : path to the ptm binary; defaults to value from config.
    """

    def __init__(
        self,
        run_dir: str,
        ptm_executable: Optional[str] = None,
    ) -> None:
        self.run_dir = str(Path(run_dir).resolve())
        if ptm_executable is not None:
            self.ptm_executable = str(Path(ptm_executable).resolve())
        else:
            cfg = load_config()
            self.ptm_executable = str(Path(cfg.ptm_executable).resolve())

        # Ensure output directory exists
        Path(self.run_dir, "ptm_output").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Single run
    # ------------------------------------------------------------------

    def run_single(
        self,
        run_id: int,
        timeout: Optional[float] = None,
        stdout_callback: Optional[Callable[[str], None]] = None,
    ) -> subprocess.CompletedProcess:
        """Run PTM for a single run_id.

        PTM is invoked as:   <ptm_executable> <run_id>
        from within run_dir.

        Parameters
        ----------
        run_id          : integer run identifier
        timeout         : optional wall-clock timeout in seconds
        stdout_callback : if provided, called with each line of stdout as PTM runs

        Returns
        -------
        subprocess.CompletedProcess with returncode, stdout, stderr.
        """
        from ptm_python.ptm_tools import cd

        cmd = [self.ptm_executable, str(run_id)]

        with cd(self.run_dir):
            if stdout_callback is None:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
            else:
                lines_out = []
                lines_err = []
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                try:
                    for line in proc.stdout:
                        stdout_callback(line.rstrip())
                        lines_out.append(line)
                    proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                    raise
                result = subprocess.CompletedProcess(
                    args=cmd,
                    returncode=proc.returncode,
                    stdout="".join(lines_out),
                    stderr="".join(proc.stderr.readlines()),
                )

        if result.returncode != 0:
            raise RuntimeError(
                f"PTM run {run_id} failed (returncode={result.returncode}).\n"
                f"stderr:\n{result.stderr}"
            )
        return result

    # ------------------------------------------------------------------
    # Parallel runs
    # ------------------------------------------------------------------

    def run_parallel(
        self,
        run_ids: List[int],
        max_workers: int = 4,
        timeout: Optional[float] = None,
    ) -> Dict[int, subprocess.CompletedProcess]:
        """Launch up to max_workers PTM processes concurrently.

        Each run_id is an independent OS subprocess, so ThreadPoolExecutor
        is appropriate here (no shared memory hazard for subprocesses).

        Parameters
        ----------
        run_ids     : list of integer run IDs
        max_workers : maximum concurrent processes
        timeout     : per-run timeout in seconds

        Returns
        -------
        {run_id: CompletedProcess} dict
        """
        results: Dict[int, subprocess.CompletedProcess] = {}
        errors: Dict[int, Exception] = {}

        def _run(run_id: int):
            return run_id, self.run_single(run_id, timeout=timeout)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_run, rid): rid for rid in run_ids}
            for future in futures:
                rid = futures[future]
                try:
                    _, result = future.result()
                    results[rid] = result
                except Exception as exc:
                    errors[rid] = exc

        if errors:
            msg = "\n".join(f"  run {rid}: {exc}" for rid, exc in errors.items())
            raise RuntimeError(f"Some PTM runs failed:\n{msg}")

        return results

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def check_output_exists(self, run_id: int) -> bool:
        """Return True if ptm_output/ptm_{run_id:04d}.dat exists."""
        path = Path(self.run_dir) / "ptm_output" / f"ptm_{run_id:04d}.dat"
        return path.is_file()

    def output_path(self, run_id: int) -> Path:
        """Return the expected trajectory output path for run_id."""
        return Path(self.run_dir) / "ptm_output" / f"ptm_{run_id:04d}.dat"

    def map_path(self, run_id: int) -> Path:
        """Return the expected flux map output path for run_id."""
        return Path(self.run_dir) / "ptm_output" / f"map_{run_id:04d}.dat"
