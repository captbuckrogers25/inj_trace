"""Tests for inj_trace.visualization modules."""

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for CI


class TestEquatorial:
    def test_plot_equatorial_bfield(self, simple_grid, tmp_path):
        from inj_trace.visualization.equatorial import plot_equatorial_bfield
        ax = plot_equatorial_bfield(simple_grid, component="bmag",
                                    save_path=str(tmp_path / "test_eq.png"))
        assert ax is not None
        assert (tmp_path / "test_eq.png").is_file()

    def test_plot_equatorial_flux(self, tmp_path):
        from inj_trace.visualization.equatorial import plot_equatorial_flux
        rng = np.random.default_rng(42)
        positions = rng.uniform(-8, -3, (50, 3))
        positions[:, 2] = 0.0
        flux = rng.lognormal(0, 1, 50)
        ax = plot_equatorial_flux(positions, flux, energy_kev=100.0,
                                   save_path=str(tmp_path / "flux.png"))
        assert ax is not None


class TestLshell:
    def test_plot_lshell_map(self, tmp_path):
        from inj_trace.visualization.lshell import plot_lshell_map
        rng = np.random.default_rng(42)
        lstar = rng.uniform(3, 7, 50)
        flux  = rng.lognormal(0, 1, 50)
        ax = plot_lshell_map(lstar, flux, energy_kev=100.0,
                              save_path=str(tmp_path / "lshell.png"))
        assert ax is not None

    def test_plot_lshell_map_with_nans(self, tmp_path):
        from inj_trace.visualization.lshell import plot_lshell_map
        lstar = np.array([4.0, 5.0, np.nan, 6.0, np.nan])
        flux  = np.array([1e3, 2e3, 1.5e3, 0.5e3, 3e3])
        ax = plot_lshell_map(lstar, flux)
        assert ax is not None


class TestTimeseries:
    def test_plot_flux_timeseries_1d(self, tmp_path):
        from inj_trace.visualization.timeseries import plot_flux_timeseries
        times = np.arange(10) * 300.0
        flux  = np.logspace(2, 4, 10)
        ax = plot_flux_timeseries(times, flux, save_path=str(tmp_path / "ts.png"))
        assert ax is not None

    def test_plot_flux_timeseries_2d(self, tmp_path):
        from inj_trace.visualization.timeseries import plot_flux_timeseries
        times = np.arange(10) * 300.0
        flux  = np.random.lognormal(0, 1, (10, 5))
        energies = np.logspace(0, 2, 5)
        ax = plot_flux_timeseries(times, flux, energies_kev=energies)
        assert ax is not None


class TestTrajectories3D:
    def test_plot_trajectory_3d(self, tmp_path):
        from inj_trace.visualization.trajectories3d import plot_trajectory_3d
        from inj_trace.postprocess.trajectories import TrajectoryData
        # Build a fake TrajectoryData
        rng = np.random.default_rng(42)
        n_pts = 20
        track = np.zeros((n_pts, 8))
        track[:, 0] = np.linspace(0, 3600, n_pts)    # time
        track[:, 1] = np.linspace(-8, -5, n_pts)      # x
        track[:, 2] = rng.normal(0, 0.5, n_pts)        # y
        track[:, 3] = rng.normal(0, 0.5, n_pts)        # z
        track[:, 6] = np.linspace(100, 200, n_pts)     # energy
        track[:, 7] = 85.0 * np.ones(n_pts)            # pitch angle
        raw = {1: track, 2: track + rng.normal(0, 0.1, track.shape)}
        traj = TrajectoryData(raw)
        ax = plot_trajectory_3d(traj, save_path=str(tmp_path / "traj.png"))
        assert ax is not None


class TestAnimation:
    def test_animate_equatorial_no_save(self):
        from inj_trace.visualization.animation import animate_equatorial_flux
        rng = np.random.default_rng(42)
        positions = rng.uniform(-8, -3, (30, 3))
        positions[:, 2] = 0.0
        flux_snaps = [rng.lognormal(0, 1, 30) for _ in range(5)]
        anim = animate_equatorial_flux(
            flux_snaps, positions, np.arange(5) * 300.0,
            save_path=None,
        )
        assert anim is not None
