# Working on Xtrack

Xtrack is the Xsuite library for particle tracking and accelerator optics on
CPU and GPU. Keep changes focused, preserve established physics conventions,
and follow the patterns in the surrounding code.

## Repository map

- `xtrack/line.py`, `environment.py`, `tracker.py`, and `tracker_data.py`:
  lattice construction, shared elements and variables, and tracking orchestration.
- `xtrack/beam_elements/`: Python element definitions; `elements_src/` contains
  their C tracking implementations.
- `xtrack/headers/` and `xtrack/particles/`: shared tracking code and particle
  data structures, including the `LocalParticle` interface.
- `xtrack/twiss/`, `match.py`, `targets.py`, and `survey/`: optics, matching,
  and geometry.
- `tests/`: pytest tests. `test_data/` contains reference lattices and physics
  data. `examples/` contains runnable examples and exploratory scripts.
- `ducktrack/`: reference tracking implementations used by some tests.

Xobjects supplies contexts, buffers, data structures, and kernel compilation;
Xdeps supplies expressions and dependencies. Tests can also depend on Xpart,
Xfields, and other Xsuite packages. The main documentation and shared CI
workflows live in the separate `xsuite` repository.

## Local workflow

- Inspect the working tree before editing. Preserve unrelated changes,
  untracked experiments, and simulation output; do not clean them up as part
  of an unrelated task.
- Use the existing Python environment when possible. Check which checkout is
  imported, especially when sibling repositories or multiple worktrees exist:
  `python -c "import xtrack; print(xtrack.__file__)"`.
- Read `pyproject.toml` for supported Python versions and dependencies. When
  setup is needed, install this checkout with `python -m pip install -e .`;
  `python -m pip install -e '.[tests]'` includes the declared test extras.
  Some tests need additional sibling packages or external programs.
- Keep changes to sibling repositories explicit and report them separately.
- Follow local formatting. Avoid broad reformatting, incidental refactoring,
  dependency changes, and version bumps outside the task's scope.

## Implementation guidance

- Trace changes through both the Python and C layers. Element fields,
  `_xofields`, generated accessors, kernel source dependencies, and serialization
  must remain consistent.
- Preserve CPU, CUDA, and OpenCL compatibility. Follow existing `GPUFUN`,
  `LocalParticle`, and context-array conventions; avoid introducing host-only
  operations into device code.
- Preserve public defaults, units, signs, coordinate conventions, and
  `to_dict`/`from_dict` behavior unless the task calls for changing them.
  Check expression-backed fields and shared elements when changing setters
  or construction logic.
- For physics changes, state the expected behavior and validate it against
  an analytic result, an independent reference, or a relevant invariant.
  Consider applicable limits, backtracking, slicing, and radiation behavior.
- Explain numerical tolerances. Do not loosen tolerances or regenerate
  reference data merely to make a failing test pass.
- Update relevant examples and API documentation when user-facing behavior
  changes. Preserve existing copyright and license headers.

## Testing

Run commands from the repository root using the selected Python environment.
Start with the affected test or module, then expand to related tests according
to the scope of the change. For example:

```sh
XOBJECTS_TEST_CONTEXTS=ContextCpu XSUITE_ALLOW_KERNEL_COMPILATION=1 \
    python -m pytest tests/test_elements.py -k test_constructor -q
```

`XOBJECTS_TEST_CONTEXTS` controls the contexts selected by Xobjects test
helpers. Use `ContextCpu` for an initial CPU check; test other relevant
contexts when available and report which were exercised.

`XSUITE_ALLOW_KERNEL_COMPILATION=1` permits compilation when a compatible
prebuilt kernel is unavailable. When validating changes to C sources, use
`XSUITE_FORCE_KERNEL_COMPILATION=1` to exercise the changed sources instead
of loading a prebuilt kernel. Tests specifically checking prebuilt-kernel
selection need their intended settings.

- Add a focused regression test for a bug fix or changed behavior. Prefer
  small, deterministic cases and existing fixtures and reference data.
- Use `xobjects.test_helpers.for_all_test_contexts` and the supplied
  `test_context` for tests intended to exercise multiple backends.
- Tests that rely on Matplotlib must be skipped when it is unavailable.
  Use `pytest.importorskip('matplotlib')` in the affected test or fixture,
  before importing plotting modules. Skip an entire module only if all its
  tests require Matplotlib.
- Respect the buffer-lifetime checks in `tests/conftest.py`. Use its temporary
  default-context fixtures where fixture lifetimes require them; investigate
  unexpected retained buffers rather than disabling the checks.
- Report test commands, results, and any missing dependencies or unavailable
  backends. Distinguish environment failures from regressions.
- For documentation-only edits, check the content and diff; running the
  tracking test suite is unnecessary.

Before finishing, review the diff for unintended files and whitespace errors.
Summarize the behavior changed, validation performed, and remaining limitations.
