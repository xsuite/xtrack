import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xtrack as xt

FIT_PARS_INDEX_COLS = [
    "field_component",
    "derivative_x",
    "region_name",
    "s_start",
    "s_end",
    "idx_start",
    "idx_end",
    "param_index",
]


def make_segment_field(evaluate_B, params_1d, multipole_order_local):
    params_arr = np.asarray(params_1d, dtype=float)

    def field(x, y, z):
        return evaluate_B(x, y, z, params_arr, multipole_order_local)

    return field


def make_test_particle(p_ref, n_particles=1):
    p = p_ref.copy()
    p.x = np.full(n_particles, 1e-3)
    p.px = np.full(n_particles, 1e-4)
    p.y = np.full(n_particles, 0.5e-3)
    p.py = np.full(n_particles, -0.5e-4)
    return p


def benchmark_line(line, p_ref, n_particles=1, n_repeats=7, n_warmup=1):
    # First-call timing includes one-time setup/cache effects.
    p_first = make_test_particle(p_ref, n_particles=n_particles)
    t0 = time.perf_counter()
    line.track(p_first)
    first_call_s = time.perf_counter() - t0

    # Warm-up before steady-state timing.
    for _ in range(n_warmup):
        line.track(make_test_particle(p_ref, n_particles=n_particles))

    times = []
    for _ in range(n_repeats):
        p = make_test_particle(p_ref, n_particles=n_particles)
        t0 = time.perf_counter()
        line.track(p)
        times.append(time.perf_counter() - t0)

    return {
        "first_call_s": float(first_call_s),
        "median_s": float(np.median(times)),
    }


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent.parent
    eval_module_path = (
        base_dir / "xtrack" / "beam_elements" / "elements_src" / "spline_B_field_eval_python.py"
    )
    spec = importlib.util.spec_from_file_location(
        "spline_B_field_eval_python", eval_module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    evaluate_B = module.evaluate_B

    df = pd.read_csv(
        base_dir / "test_data" / "sls" / "undulator_fit_pars.csv",
        index_col=FIT_PARS_INDEX_COLS,
    )
    multipole_order = 3

    # Build line using piecewise SplineBorisSequence (baseline xtrack implementation).
    seq = xt.SplineBorisSequence(
        df_fit_pars=df,
        multipole_order=multipole_order,
        steps_per_point=1,
    )
    line_spline = seq.to_line()

    # Build a line with BorisSpatialIntegrator, using the same fitted coefficients.
    boris_elems = []
    for elem in seq.elements:
        params_i = np.asarray(elem.par_table, dtype=float)
        field_i = make_segment_field(evaluate_B, params_i, multipole_order)
        boris_elems.append(
            xt.BorisSpatialIntegrator(
                fieldmap_callable=field_i,
                s_start=float(elem.s_start),
                s_end=float(elem.s_end),
                n_steps=int(elem.n_steps),
            )
        )
    line_boris = xt.Line(elements=boris_elems)

    p_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1, p0c=2.7e9)
    line_spline.particle_ref = p_ref.copy()
    line_boris.particle_ref = p_ref.copy()

    n_particles = 1
    n_repeats = 7
    n_warmup = 1

    # Check that both methods give consistent end coordinates.
    p_spline = make_test_particle(p_ref, n_particles=n_particles)
    p_boris = make_test_particle(p_ref, n_particles=n_particles)
    line_spline.track(p_spline)
    line_boris.track(p_boris)

    np.testing.assert_allclose(p_spline.x, p_boris.x, rtol=1e-12, atol=5e-11)
    np.testing.assert_allclose(p_spline.px, p_boris.px, rtol=1e-12, atol=5e-11)
    np.testing.assert_allclose(p_spline.y, p_boris.y, rtol=1e-12, atol=5e-11)
    np.testing.assert_allclose(p_spline.py, p_boris.py, rtol=1e-12, atol=5e-11)

    # Benchmark both methods with exactly the same protocol.
    t_spline = benchmark_line(
        line_spline, p_ref, n_particles=n_particles, n_repeats=n_repeats, n_warmup=n_warmup
    )
    t_boris = benchmark_line(
        line_boris, p_ref, n_particles=n_particles, n_repeats=n_repeats, n_warmup=n_warmup
    )

    print(
        f"Benchmark settings: n_particles={n_particles}, "
        f"n_warmup={n_warmup}, n_repeats={n_repeats}"
    )
    print(f"SplineBorisSequence first call: {t_spline['first_call_s']:.6f} s")
    print(f"BorisSpatialIntegrator first call: {t_boris['first_call_s']:.6f} s")
    print(f"Speedup first call (seq vs boris): {t_boris['first_call_s'] / t_spline['first_call_s']:.2f}x")
    print("")
    print(f"SplineBorisSequence steady-state: {t_spline['median_s']:.6f} s (median)")
    print(f"BorisSpatialIntegrator steady-state: {t_boris['median_s']:.6f} s (median)")
    print(f"Speedup steady-state (seq vs boris): {t_boris['median_s'] / t_spline['median_s']:.2f}x")
