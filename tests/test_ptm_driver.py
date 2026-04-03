"""Tests for inj_trace.runner modules."""

import pytest
import numpy as np
from pathlib import Path


class TestPTMRunConfig:
    @pytest.mark.needs_ptm
    def test_write_defaults(self, tmp_path):
        from inj_trace.runner.ptm_setup import PTMRunConfig
        cfg = PTMRunConfig(run_id=1, n_particles=10, tlo=0, thi=3600, n_snapshots=2)
        cfg.set_electron_defaults()
        cfg.set_spatial_distribution(2, xmin=-7, xmax=-6, ymin=-0.5, ymax=0.5,
                                     zmin=-0.5, zmax=0.5)
        cfg.set_velocity_distribution(3, emin=1, emax=100, nenergy=5, npitch=5,
                                      pamin=0, pamax=90, xsource=-12.0)
        cfg.write(str(tmp_path))

        param_file = tmp_path / "ptm_parameters_0001.txt"
        dens_file  = tmp_path / "dist_density_0001.txt"
        vel_file   = tmp_path / "dist_velocity_0001.txt"
        assert param_file.is_file()
        assert dens_file.is_file()
        assert vel_file.is_file()

        # Check charge=-1 (electron)
        lines = param_file.read_text().splitlines()
        charge_line = [l for l in lines if "charge" in l][0]
        assert "-1" in charge_line

    @pytest.mark.needs_ptm
    def test_electron_defaults(self, tmp_path):
        from inj_trace.runner.ptm_setup import PTMRunConfig
        cfg = PTMRunConfig(run_id=2, tlo=0, thi=600)
        cfg.set_electron_defaults()
        cfg.set_spatial_distribution(1, x0=-5, y0=0, z0=0)
        cfg.set_velocity_distribution(1, ekev=100, alpha=90, phi=180)
        cfg.write(str(tmp_path))
        pf = tmp_path / "ptm_parameters_0002.txt"
        content = pf.read_text()
        assert "charge" in content
        assert "iswitch" in content


class TestPTMExecutor:
    def test_output_path(self, tmp_path):
        from inj_trace.runner.executor import PTMExecutor
        exe = PTMExecutor.__new__(PTMExecutor)
        exe.run_dir = str(tmp_path)
        exe.ptm_executable = "/nonexistent/ptm"
        assert str(exe.output_path(1)).endswith("ptm_0001.dat")
        assert str(exe.map_path(3)).endswith("map_0003.dat")

    def test_check_output_exists(self, tmp_path):
        from inj_trace.runner.executor import PTMExecutor
        exe = PTMExecutor.__new__(PTMExecutor)
        exe.run_dir = str(tmp_path)
        exe.ptm_executable = "/nonexistent/ptm"
        assert not exe.check_output_exists(1)

        # Create the expected file
        out_dir = tmp_path / "ptm_output"
        out_dir.mkdir()
        (out_dir / "ptm_0001.dat").write_text("# 1\n0.0 -5 0 0 0 0 100 90\n")
        assert exe.check_output_exists(1)
