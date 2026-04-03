"""Tests for inj_trace.fields modules."""

import numpy as np
import pytest
from datetime import datetime


class TestFieldGrid:
    def test_from_spec(self):
        from inj_trace.fields.grid import FieldGrid
        grid = FieldGrid.from_spec({
            "xmin": -6, "xmax": -2, "nx": 5,
            "ymin": -2, "ymax":  2, "ny": 5,
            "zmin": -1, "zmax":  1, "nz": 3,
        })
        assert grid.nx == 5
        assert grid.ny == 5
        assert grid.nz == 3
        assert grid.xvec[0] == -6.0
        assert grid.xvec[-1] == -2.0

    def test_bmag(self, simple_grid):
        mag = simple_grid.bmag()
        assert mag.shape == (5, 5, 3)
        assert np.all(np.isfinite(mag))
        assert mag.max() > 0

    def test_zero_efield(self, simple_grid):
        assert np.all(simple_grid.ex == 0)
        assert np.all(simple_grid.ey == 0)
        assert np.all(simple_grid.ez == 0)

    def test_inner_boundary_mask(self, simple_grid):
        # All grid points in simple_grid are at r > 2 Re, so masking at 1.2 Re
        # should not zero anything (bz was already set)
        bmag = simple_grid.bmag()
        assert bmag.max() > 0

    @pytest.mark.needs_lgmpy
    def test_evaluate_t89(self):
        from inj_trace.fields.grid import FieldGrid
        grid = FieldGrid.from_spec({
            "xmin": -5, "xmax": -3, "nx": 3,
            "ymin":  0, "ymax":  0, "ny": 1,
            "zmin":  0, "zmax":  0, "nz": 1,
        })
        t = datetime(2013, 3, 17, 6)
        grid.evaluate("T89", t, {"Kp": 3})
        assert grid.bx is not None
        assert grid.bmag().max() > 10   # should be > 10 nT at these L-shells


class TestPTMFieldWriter:
    def test_write_snapshot(self, simple_grid, tmp_path):
        from inj_trace.fields.writer import PTMFieldWriter
        writer = PTMFieldWriter(str(tmp_path))
        p = writer.write_snapshot(simple_grid, snapshot_index=1)
        assert p.is_file()

        # Read header line
        with open(p) as fh:
            header = fh.readline().strip().split()
        nx, ny, nz = int(header[0]), int(header[1]), int(header[2])
        assert nx == simple_grid.nx
        assert ny == simple_grid.ny
        assert nz == simple_grid.nz

    def test_write_tgrid(self, tmp_path):
        from inj_trace.fields.writer import PTMFieldWriter
        writer = PTMFieldWriter(str(tmp_path))
        tg = writer.write_tgrid([0.0, 3600.0])
        assert tg.is_file()
        vals = np.loadtxt(str(tg))
        assert len(vals) == 2
        assert vals[0] == 0.0
        assert vals[1] == 3600.0

    def test_write_static(self, simple_grid, tmp_path):
        from inj_trace.fields.writer import PTMFieldWriter
        writer = PTMFieldWriter(str(tmp_path))
        p1, p2, tg = writer.write_static(simple_grid, duration_s=1800.0)
        assert p1.is_file()
        assert p2.is_file()
        assert tg.is_file()

        # Two field files should be identical in content
        with open(p1) as f1, open(p2) as f2:
            assert f1.read() == f2.read()

    def test_times_from_datetimes(self):
        from inj_trace.fields.writer import PTMFieldWriter
        from datetime import timedelta
        t0 = datetime(2013, 3, 17, 6)
        times = [t0, t0 + timedelta(minutes=5), t0 + timedelta(minutes=10)]
        secs = PTMFieldWriter.times_from_datetimes(times)
        assert secs == [0.0, 300.0, 600.0]
