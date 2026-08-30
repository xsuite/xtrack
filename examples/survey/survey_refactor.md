# Survey refactor: `Frame`

## Goal

Refactor Xtrack survey propagation around a mutable `Frame` object while
preserving the `Line.survey` API and its numerical conventions.

`Frame` will own all operations that move or rotate a local reference frame.
Both the main survey loop and element-specific survey hooks will propagate the
survey by mutating a `Frame` in place.

## Frame model

`Frame` stores a homogeneous 4 x 4 matrix that maps local survey coordinates
`(x, y, s, 1)` to global coordinates `(X, Y, Z, 1)`:

- `matrix[:3, :3]` is the orientation matrix currently called `E_matrix`.
- `matrix[:3, 3]` is the position currently called `XYZ`.
- Transformations act in the local frame by post-multiplying the matrix.
- Angles are expressed in radians, consistently with the survey API.
- Transformation methods mutate the frame in place and return `self`, so they
  can be chained.
- Propagation must not silently re-orthogonalize the matrix, since that could
  change existing numerical results.

Planned construction and access API:

```python
Frame()
Frame(matrix)
Frame.from_xyz_angles(X, Y, Z, theta, phi, psi)
Frame.from_xyz_matrix(XYZ, E_matrix)

frame.matrix
frame.xyz
frame.rotation
frame.copy()
frame.inverse()
```

Planned propagation methods:

```python
frame.trans_x(dx)
frame.trans_y(dy)
frame.trans_s(ds)

frame.rot_x(angle)
frame.rot_y(angle)
frame.rot_s(angle)

frame.arc_x(length, angle)
frame.arc_y(length, angle)
frame.arc(length, angle, tilt=0)
```

`arc` is the canonical implementation of the present MAD-X-compatible bend
propagation. `arc_x` and `arc_y` are convenience operations. Their sign
conventions will be fixed by tests reproducing the present survey behavior
before the old implementation is replaced.

The class is expected to live in `xtrack/frame.py` and to be publicly
available as `xtrack.Frame`, since it is part of the custom-element protocol.

## Element protocol

The new hook is:

```python
def track_frame(self, frame, backtrack=False):
    ...
```

The element mutates `frame` in place by calling its transformation methods.
No returned frame is required.

`backtrack` is part of the protocol because the inverse of a composite
transformation can require reversing the order of its operations. For
example, backtracking through `Rotation(seq='yxs')` cannot in general be
implemented by the survey loop merely negating three angles.

Survey hook dispatch order:

1. Use `track_frame` when the element exposes it.
2. Otherwise apply the standard drift or bend propagation in the survey loop.

There is no legacy hook adapter. Custom elements that need non-standard survey
propagation must implement `track_frame`.

## Retained interface and cleanup decision

The following existing interfaces and behavior must remain unchanged:

- `Line.survey(...)`, including its signature and returned `SurveyTable`.
- Survey columns, scalars, names, ordering, shapes, and dtypes.
- Surveys initialized at the start or at an intermediate `element0`.
- Forward and backward propagation used for intermediate initialization.
- `include_element_frames=True` and survey-table reversal.
- `get_survey(...)`.

The compatibility-only helpers `advance_element`, `advance_bend`,
`advance_rotation`, `advance_drift`, `get_E_from_angles`,
`get_angles_from_w`, and `compute_survey` are intentionally removed. The
`survey_advance_element` beam-elements import and legacy `_propagate_survey`
hook are removed with them. Survey propagation uses `Frame` directly, without
conversion to and from `(v, w)`.

## Scope of transformation centralization

All survey-frame movement and rotation belongs in `Frame`. This includes:

- Standard drift and bend propagation in `survey.py`.
- Translation and rotation elements.
- Thick and drift slices.
- Straight-body RBend entry and exit maps.
- Physical element-frame construction in `misalignment_survey.py`.
- Relative frame composition and inversion where practical.

Misalignment helpers may continue to calculate scalar transformation
parameters. Creation, composition, inversion, and application of homogeneous
frame transforms should use `Frame`, rather than introducing another survey
propagation implementation.

## Implementation plan

### 1. Freeze current behavior and add `Frame`

- [x] Add independent reference tests for the current translation, rotation,
      drift, bend, tilt, and zero-length bend formulas.
- [x] Implement the homogeneous matrix representation and constructors.
- [x] Implement position and orientation accessors.
- [x] Implement local translations and rotations.
- [x] Implement arc propagation with the existing MAD-X sign conventions.
- [x] Implement copy, composition, and rigid inverse operations needed by the
      survey code.
- [x] Test chaining and forward/inverse round trips from non-identity frames.

### 2. Refactor the main survey loop

- [x] Initialize a `Frame` from the existing survey initial coordinates and
      angles.
- [x] Store copies of its position and orientation at each element entrance.
- [x] Use `Frame` for default drift and bend propagation.
- [x] Preserve the current split forward/backward logic for `element0`.
- [x] Keep `get_survey` return values and types compatible.
- [x] Remove the old procedural survey helper APIs.

### 3. Introduce the `track_frame` hook

- [x] Dispatch to `track_frame` when exposed by an element.
- [x] Test a custom element implementing only `track_frame`.

### 4. Migrate built-in elements

- [x] `Translation` and `XYShift`.
- [x] `XRotation`, `YRotation`, and `SRotation`.
- [x] General `Rotation`, including reversed sequence during backtracking.
- [x] Thick slices and drift slices.
- [x] Straight-body RBend entry and exit edge slices.
- [x] Remove `survey_advance_element` and built-in dependence on the old
      procedural propagation helpers.

### 5. Refactor element-frame and misalignment helpers

- [x] Use `Frame` in `get_misaligned_element_survey`.
- [x] Use `Frame` in RBend physical-frame helpers.
- [x] Replace direct propagation through temporary `Translation` and
      `Rotation` elements with direct frame operations.
- [x] Preserve straight and curved transformation order exactly.
- [x] Preserve sliced-element anchor and weight handling.
- [x] Use `Frame` for relative survey transform construction where this
      reduces duplicate homogeneous-matrix logic.

### 6. Verification and documentation

- [x] Run focused `Frame` and compatibility-hook tests.
- [x] Run `tests/test_survey.py` on the available serial CPU context.
- [x] Run relevant thick-element, slicing, transformation, and RBend tests.
- [x] Run MAD-X survey comparisons when their optional dependencies and data
      are available.
- [x] Compare representative survey tables before and after the refactor at
      tight absolute tolerance.
- [x] Run the wider Xtrack test suite in proportion to the affected scope.
- [x] Document `Frame` and the custom-element `track_frame` protocol.

Development environment:

```text
/Users/giadarol/miniforge3/envs/py313/bin/python
```

## Acceptance criteria

- Existing public survey calls require no user changes.
- Existing survey regression tests pass without relaxed tolerances unless a
  difference is understood and explicitly accepted.
- A custom `track_frame` element can express its geometry using only `Frame`
  methods.
- Built-in element survey methods contain no standalone matrix propagation
  formulas.
- Survey-frame transformation formulas have one implementation in `Frame`.
- Forward and intermediate-element surveys give equivalent positions and
  orientations to the pre-refactor implementation.

## Progress log

- 2026-08-30: Design agreed. Plan recorded. No implementation started.
- 2026-08-30: Added `Frame`, refactored the survey loop and compatibility
  helpers, migrated built-in hooks, and centralized element-frame transforms.
- 2026-08-30: Focused Frame, survey, slicing, misalignment, thick-element,
  native-loader, and MAD-X comparison tests pass. Automatic OpenMP test
  contexts cannot compile with the system Apple clang (`-fopenmp` is not
  supported); equivalent serial CPU cases pass.
- 2026-08-30: Final affected-scope regression batch completed with 121 passed
  and 61 deselected automatic/non-serial contexts.
- 2026-08-30: Backward compatibility for `_propagate_survey` and the old
  procedural survey helpers was intentionally dropped to remove duplication.
- 2026-08-30: Post-cleanup affected-scope regression batch completed with 123
  passed and 61 automatic/non-serial contexts deselected.
