# Survey refactor: `Frame`

## Goal

Refactor Xtrack survey propagation around a mutable `Frame` object without
changing the existing external survey API or its numerical conventions.

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

1. Use the most-specific `track_frame` implementation.
2. Otherwise adapt the most-specific legacy `_propagate_survey` implementation.
3. Otherwise apply the standard drift or bend propagation in the survey loop.

The dispatcher must consider the class MRO, not only `hasattr`. In particular,
a user subclass may override `_propagate_survey` while inheriting
`track_frame` from a built-in parent. The more-specific user override must
continue to win. If both hooks are defined by the same class, `track_frame`
takes precedence.

Legacy `_propagate_survey(v, w, backtrack)` support will remain silent, without
a new deprecation warning. Built-in elements will migrate to `track_frame`.

## Compatibility requirements

The following existing interfaces and behavior must remain unchanged:

- `Line.survey(...)`, including its signature and returned `SurveyTable`.
- Survey columns, scalars, names, ordering, shapes, and dtypes.
- Surveys initialized at the start or at an intermediate `element0`.
- Forward and backward propagation used for intermediate initialization.
- `include_element_frames=True` and survey-table reversal.
- `get_survey(...)`.
- `advance_element(v, w, ...)`.
- `get_E_from_angles(...)` and `get_angles_from_w(...)`.
- `compute_survey(...)` and its existing deprecation behavior.
- Existing custom elements that expose only `_propagate_survey`.
- Existing `xtrack.beam_elements.survey_advance_element` availability.

The procedural helpers can remain as compatibility wrappers around `Frame`.
The main survey loop and migrated built-in elements should use `Frame`
directly, avoiding conversion to and from `(v, w)` at every element.

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

- [ ] Add independent reference tests for the current translation, rotation,
      drift, bend, tilt, and zero-length bend formulas.
- [ ] Implement the homogeneous matrix representation and constructors.
- [ ] Implement position and orientation accessors.
- [ ] Implement local translations and rotations.
- [ ] Implement arc propagation with the existing MAD-X sign conventions.
- [ ] Implement copy, composition, and rigid inverse operations needed by the
      survey code.
- [ ] Test chaining and forward/inverse round trips from non-identity frames.

### 2. Refactor the main survey loop

- [ ] Initialize a `Frame` from the existing survey initial coordinates and
      angles.
- [ ] Store copies of its position and orientation at each element entrance.
- [ ] Use `Frame` for default drift and bend propagation.
- [ ] Preserve the current split forward/backward logic for `element0`.
- [ ] Keep `get_survey` return values and types compatible.
- [ ] Reimplement `advance_element` as a compatibility wrapper.

### 3. Introduce hook dispatch and legacy compatibility

- [ ] Add MRO-aware selection between `track_frame` and
      `_propagate_survey`.
- [ ] Adapt legacy `(v, w)` input and output to the active `Frame`.
- [ ] Test a custom element implementing only `track_frame`.
- [ ] Test a custom element implementing only `_propagate_survey`.
- [ ] Test a subclass whose legacy override is more specific than an inherited
      `track_frame`.
- [ ] Test that `track_frame` wins when both hooks are defined at the same
      class level.

### 4. Migrate built-in elements

- [ ] `Translation` and `XYShift`.
- [ ] `XRotation`, `YRotation`, and `SRotation`.
- [ ] General `Rotation`, including reversed sequence during backtracking.
- [ ] Thick slices and drift slices.
- [ ] Straight-body RBend entry and exit edge slices.
- [ ] Remove built-in dependence on `survey_advance_element` where it is no
      longer needed, while preserving the public import compatibility.

### 5. Refactor element-frame and misalignment helpers

- [ ] Use `Frame` in `get_misaligned_element_survey`.
- [ ] Use `Frame` in RBend physical-frame helpers.
- [ ] Replace direct propagation through temporary `Translation` and
      `Rotation` elements with direct frame operations.
- [ ] Preserve straight and curved transformation order exactly.
- [ ] Preserve sliced-element anchor and weight handling.
- [ ] Use `Frame` for relative survey transform construction where this
      reduces duplicate homogeneous-matrix logic.

### 6. Verification and documentation

- [ ] Run focused `Frame` and compatibility-hook tests.
- [ ] Run `tests/test_survey.py`.
- [ ] Run relevant thick-element, slicing, transformation, and RBend tests.
- [ ] Run MAD-X survey comparisons when their optional dependencies and data
      are available.
- [ ] Compare representative survey tables before and after the refactor at
      tight absolute tolerance.
- [ ] Run the wider Xtrack test suite in proportion to the affected scope.
- [ ] Document `Frame` and the custom-element `track_frame` protocol.

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
- A custom legacy `_propagate_survey` element continues to work.
- Built-in element survey methods contain no standalone matrix propagation
  formulas.
- Survey-frame transformation formulas have one implementation in `Frame`.
- Forward and intermediate-element surveys give equivalent positions and
  orientations to the pre-refactor implementation.

## Progress log

- 2026-08-30: Design agreed. Plan recorded. No implementation started.
