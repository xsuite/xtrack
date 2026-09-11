# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

# MADX Reference:
# https://github.com/MethodicalAcceleratorDesign/MAD-X/blob/2dcd046b1f6ca2b44ef67c8d572ff74370deee25/src/survey.f90


import numpy as np

from .frame import Frame, _angles_from_E_matrix
from ..table import Table
from ..general import DEPRECATION_INFO_PREP_1_0


__all__ = [
    'SurveyTable',
    'get_survey',
    'survey_from_line',
    'survey_relative_transform',
    'track_frame',
]


_ELEMENT_FRAME_NAMES = (
    'ref_start',
    'ref_end',
    'elem_start',
    'elem_end',
)


class SurveyTable(Table):
    """
    Table for survey data.

    ``SurveyTable`` stores the surveyed position and orientation of each
    element along a line. Typical columns include the longitudinal position
    ``s``, global coordinates such as ``X``, ``Y``, and ``Z``, and orientation
    data such as the local reference-frame basis vectors or rotation matrices.
    """

    def __init__(self, data, *args, **kwargs):
        """
        Create a survey table.

        Parameters
        ----------
        data : mapping
            Mapping containing survey-table columns. Typical columns include
            ``name``, ``element_type``, ``s``, ``X``, ``Y``, ``Z``, and
            orientation data.
        *args
            Additional positional arguments passed to :class:`xtrack.Table`.
        **kwargs
            Additional keyword arguments passed to :class:`xtrack.Table`.

        Examples
        --------
        Build a compact survey table:

        >>> import numpy as np
        >>> from xtrack.survey import SurveyTable
        >>> tab = SurveyTable({
        ...     "name": np.array(["mqf.1", "d1.1", "mb1.1", "_end_point"],
        ...                      dtype=object),
        ...     "element_type": np.array(["Quadrupole", "Drift", "Bend", ""],
        ...                              dtype=object),
        ...     "s": np.array([0.0, 0.3, 1.3, 4.3]),
        ...     "X": np.array([0.0, 0.0, 0.0, 3.0]),
        ...     "Y": np.array([0.0, 0.0, 0.0, 0.0]),
        ...     "Z": np.array([0.0, 0.3, 1.3, 3.7]),
        ... })
        >>> tab
        SurveyTable: 4 rows, 6 cols
        name       element_type             s             X             Y             Z
        mqf.1      Quadrupole               0             0             0             0
        d1.1       Drift                  0.3             0             0           0.3
        mb1.1      Bend                   1.3             0             0           1.3
        _end_point                        4.3             3             0           3.7

        Select coordinates or a longitudinal range:

        >>> tab.cols["s X Z"]
        SurveyTable: 4 rows, 4 cols
        name                   s             X             Z
        mqf.1                  0             0             0
        d1.1                 0.3             0           0.3
        mb1.1                1.3             0           1.3
        _end_point           4.3             3           3.7
        >>> tab.rows[0.0:1.5:"s"]
        SurveyTable: 3 rows, 6 cols
        name  element_type             s             X             Y             Z
        mqf.1 Quadrupole               0             0             0             0
        d1.1  Drift                  0.3             0             0           0.3
        mb1.1 Bend                   1.3             0             0           1.3
        """
        super().__init__(data, *args, **kwargs)

    _DEPRECATED_FIELDS = {
        'p0': ('`p0` is deprecated, please use `XYZ` instead'
                      + DEPRECATION_INFO_PREP_1_0),
        'W': ('`W` is deprecated, please use `E_matrix` instead'
                      + DEPRECATION_INFO_PREP_1_0)
    }

    _error_on_row_not_found = True

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('sep_count', '::::')
        super().__init__(*args, **kwargs)

    def reverse(self):
        """
        Build a survey table for the reverse local reference frame.

        The returned table has the element order reversed and survey quantities
        transformed to the reverse local reference frame. The longitudinal position
        is transformed as ``s -> line_length - s``. The global coordinates are
        transformed as ``X -> -X``, ``Y -> Y``, and ``Z -> -Z``. The survey
        orientation matrix is transformed consistently with the reversed global
        frame and reversed local frame axes.

        Returns
        -------
        xtrack.survey.SurveyTable
            Survey table corresponding to the reverse local reference frame.
        """

        new_cols = {}

        element_properties = ['name', 'element_type', 'isthick', 'drift_length',
                              'length', 'prototype']

        for kk in element_properties:
            new_cols[kk] = self._data[kk].copy()
            new_cols[kk][:-1] = new_cols[kk][:-1][::-1]
            new_cols[kk][-1] = self._data[kk][-1]

        itake = slice(1, None, None)

        # s vector
        new_cols['s'] = self._data['s'].copy()
        new_cols['s'][:-1] = new_cols['s'][itake][::-1]
        new_cols['s'][-1] = self._data['s'][0]

        new_cols['s'] = self._data['s'][-1] - new_cols['s']

        new_E_matrix = self.E_matrix.copy()
        new_E_matrix[:-1, :, :] = new_E_matrix[itake, :, :][::-1, :, :]
        new_E_matrix[-1, :, :] = self.E_matrix[0, :, :]

        new_V = self.XYZ.copy()
        new_V[:-1, :] = new_V[itake, :][::-1, :]
        new_V[-1, :] = self.XYZ[0, :]

        # Reverse X and Z in the global frame
        new_V[:, 0] *= -1
        new_V[:, 2] *= -1
        new_E_matrix[:, 0, :] *= -1
        new_E_matrix[:, 2, :] *= -1

        # Reverse ix and iy of the local frame
        new_E_matrix[:, :, 0] *= -1
        new_E_matrix[:, :, 2] *= -1

        derived_quantities = _get_survey_quantities_from_v_w(new_V, new_E_matrix)
        new_cols.update(derived_quantities)

        if 'XYZ_ref_start' in self._data:
            position_pairs = [
                ('XYZ_ref_start', 'XYZ_ref_end'),
                ('XYZ_elem_start', 'XYZ_elem_end'),
            ]
            orientation_pairs = [
                ('E_ref_start', 'E_ref_end'),
                ('E_elem_start', 'E_elem_end'),
            ]

            for start_name, end_name in position_pairs:
                reversed_start = self._data[start_name].copy()
                reversed_end = self._data[end_name].copy()
                reversed_start[:-1] = self._data[end_name][:-1][::-1]
                reversed_end[:-1] = self._data[start_name][:-1][::-1]
                reversed_start[-1] = self.XYZ[0]
                reversed_end[-1] = self.XYZ[0]
                reversed_start[:, (0, 2)] *= -1
                reversed_end[:, (0, 2)] *= -1
                new_cols[start_name] = reversed_start
                new_cols[end_name] = reversed_end

            for start_name, end_name in orientation_pairs:
                reversed_start = self._data[start_name].copy()
                reversed_end = self._data[end_name].copy()
                reversed_start[:-1] = self._data[end_name][:-1][::-1]
                reversed_end[:-1] = self._data[start_name][:-1][::-1]
                reversed_start[-1] = self.E_matrix[0]
                reversed_end[-1] = self.E_matrix[0]
                for reversed_matrix in (reversed_start, reversed_end):
                    reversed_matrix[:, (0, 2), :] *= -1
                    reversed_matrix[:, :, (0, 2)] *= -1
                new_cols[start_name] = reversed_start
                new_cols[end_name] = reversed_end

            for frame_name in (
                    'ref_start', 'ref_end', 'elem_start', 'elem_end'):
                XYZ_frame = new_cols[f'XYZ_{frame_name}']
                for ii, coordinate in enumerate('XYZ'):
                    new_cols[f'{coordinate}_{frame_name}'] = XYZ_frame[:, ii]

        out = SurveyTable(
            data        = (new_cols | {'element0': self.element0}),
            col_names   = new_cols.keys())

        return out

    def get_frame(self, at, *, which=None):
        """Return an independent frame at a surveyed location.

        Parameters
        ----------
        at : str or int
            Element name or row index.
        which : {None, 'ref_start', 'ref_end', 'elem_start', 'elem_end'}
            Frame to return. By default, use the standard ``XYZ`` and
            ``E_matrix`` survey columns. The other choices require a survey
            generated with ``include_element_frames=True``.

        Returns
        -------
        Frame
            A new frame initialized from the selected survey row.
        """
        if which is not None and which not in _ELEMENT_FRAME_NAMES:
            raise ValueError(
                f'Invalid frame {which!r}; expected one of '
                f'{_ELEMENT_FRAME_NAMES}')

        if which is None:
            XYZ_column = 'XYZ'
            E_column = 'E_matrix'
        else:
            XYZ_column = f'XYZ_{which}'
            E_column = f'E_{which}'
            if XYZ_column not in self._data or E_column not in self._data:
                raise ValueError(
                    f'Frame {which!r} is unavailable; generate the survey '
                    'with include_element_frames=True')

        if isinstance(at, str):
            at = self.rows.get_index(at)
        return Frame.from_survey(
            self._data[XYZ_column][at], self._data[E_column][at])

    def get_all_frames(self, at):
        """Return all reference and element frames at a surveyed location.

        The survey must have been generated with
        ``include_element_frames=True``.
        """
        return {
            which: self.get_frame(at, which=which)
            for which in _ELEMENT_FRAME_NAMES
        }

    def plot(self, element_width = None, legend = True, **kwargs):
        """
        Plot the survey using ``xplt.FloorPlot``.

        Parameters
        ----------
        element_width : float, optional
            Width used to draw elements in the floor plot. If not provided, a
            value is chosen from the extent of the survey.
        legend : bool, optional
            Whether to add a matplotlib legend.
        **kwargs
            Additional keyword arguments passed to ``xplt.FloorPlot``.
        """
        # Import the xplt module here
        # (Not at the top as not default installation with xsuite)
        import xplt

        # Shallow copy of self
        out_sv_table = SurveyTable.__new__(SurveyTable)
        out_sv_table.__dict__.update(self.__dict__)
        out_sv_table._data = {
            kk: (vv.copy() if hasattr(vv, 'copy') else vv)
            for kk, vv in self._data.items()
        }

        # Removing the count for repeated elements
        out_sv_table._data['name'] = np.array([nn.split('::')[0] for nn in out_sv_table._data['name']])
        out_sv_table._index_cache = None
        out_sv_table._count_cache = None
        out_sv_table._names_cache = None

        # Setting element width for plotting
        if element_width is None:
            x_range = max(self.X) - min(self.X)
            y_range = max(self.Y) - min(self.Y)
            z_range = max(self.Z) - min(self.Z)
            element_width   = max([x_range, y_range, z_range]) * 0.03

        xplt.FloorPlot(
            survey          = out_sv_table,
            line            = self.line,
            element_width   = element_width,
            **kwargs)

        if legend:
            import matplotlib.pyplot as plt
            plt.legend()

    def to_pandas(self, index=None, columns=None):
        """
        Convert the survey table to a pandas DataFrame.

        Parameters
        ----------
        index : str, optional
            Column to use as the DataFrame index.
        columns : sequence of str, optional
            Columns to include in the DataFrame. If not provided, all table
            columns are included.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the selected survey table columns.
        """
        if columns is None:
            columns = self._col_names

        data = self._data.copy()
        for cc in columns:
            if len(data[cc]) > 1:
                data[cc] = [data[cc][ii] for ii in range(len(data[cc])) if cc in self._col_names]

        import pandas as pd
        df = pd.DataFrame(data, columns=self._col_names)
        if index is not None:
            df.set_index(index, inplace=True)
        return df


# ==================================================
# Main function
# ==================================================
def survey_from_line(
        line,
        X0=0, Y0=0, Z0=0, theta0=0, phi0=0, psi0=0,
        element0=0, values_at_element_exit=False, reverse=False,
        include_element_frames=False):
    """Execute SURVEY command. Based on MAD-X equivalent.
    Attributes, must be given in this order in the dictionary:
    X0        (float)    Initial X position in meters.
    Y0        (float)    Initial Y position in meters.
    Z0        (float)    Initial Z position in meters.
    theta0    (float)    Initial azimuthal angle in radians.
    phi0      (float)    Initial elevation angle in radians.
    psi0      (float)    Initial roll angle in radians."""

    if reverse:
        raise ValueError('`survey(..., reverse=True)` not supported anymore. '
                         'Use `survey(...).reverse()` instead.')

    assert not values_at_element_exit, "Not implemented yet"

    # Get line table to extract attributes
    tt      = line.get_table(attr = True)

    # Extract angle and tilt from elements
    angle   = tt.angle
    tilt    = tt.rot_s_rad

    # Extract drift lengths
    drift_length = tt.length
    drift_length[~tt.isthick] = 0

    if isinstance(element0, str):
        element0 = line.element_names.index(element0)

    V, E_matrix = get_survey(
        elements        = line._elements,
        X0              = X0,
        Y0              = Y0,
        Z0              = Z0,
        theta0          = theta0,
        phi0            = phi0,
        psi0            = psi0,
        drift_length    = drift_length[:-1],
        angle           = angle[:-1],
        tilt            = tilt[:-1],
        element0        = element0)

    derived_quantities = _get_survey_quantities_from_v_w(V, E_matrix)

    out_columns = derived_quantities
    out_scalars = {}

    if include_element_frames:
        from .misalignment_survey import get_element_frame_columns

        out_columns.update(get_element_frame_columns(
            elements=line._elements,
            XYZ=V,
            E_matrix=E_matrix,
        ))

    # element properties
    out_columns["name"]             = tt.name
    out_columns["element_type"]     = tt.element_type
    out_columns['isthick']          = tt.isthick
    out_columns['prototype']        = tt.prototype
    out_columns['drift_length']     = drift_length
    out_columns['length']           = tt.length

    out_columns["s"]                = tt.s

    out_scalars['element0']     = element0

    out = SurveyTable(
        data        = {**out_columns, **out_scalars},  # this is a merge
        col_names   = out_columns.keys())
    out._data['line'] = line

    return out


def get_survey(
        elements,
        X0, Y0, Z0, theta0, phi0, psi0,
        drift_length, angle, tilt,
        element0 = 0, backtrack=False):
    """
    Compute survey from initial position and orientation.
    """

    # If element0 is not the first element, split the survey
    if element0 != 0:

        # Forward section of survey
        elements_forward        = elements[element0:]
        drift_forward           = drift_length[element0:]
        angle_forward           = angle[element0:]
        tilt_forward            = tilt[element0:]

        # Evaluate forward survey
        (V_forward, E_forward)    = get_survey(
            elements        = elements_forward,
            X0              = X0,
            Y0              = Y0,
            Z0              = Z0,
            theta0          = theta0,
            phi0            = phi0,
            psi0            = psi0,
            drift_length    = drift_forward,
            angle           = angle_forward,
            tilt            = tilt_forward,
            backtrack       = backtrack)

        # Backward section of survey
        elements_backward       = elements[:element0][::-1]
        drift_backward          = np.array(drift_length[:element0][::-1])
        angle_backward          = np.array(angle[:element0][::-1])
        tilt_backward           = np.array(tilt[:element0][::-1])
        # Evaluate backward survey
        (V_backward, E_backward)   = get_survey(
            elements        = elements_backward,
            X0              = X0,
            Y0              = Y0,
            Z0              = Z0,
            theta0          = theta0,
            phi0            = phi0,
            psi0            = psi0,
            drift_length    = drift_backward,
            angle           = angle_backward,
            tilt            = tilt_backward,
            element0        = 0,
            backtrack       = not backtrack)

        # Concatenate forward and backward
        E_matrix       = np.array(E_backward[::-1][:-1] + E_forward)
        V       = np.array(V_backward[::-1][:-1] + V_forward)
        return V, E_matrix

    # Initialise lists for storing the survey
    E_matrix = []
    V = []

    # Initial position and orientation
    frame = Frame.from_survey_angles(
        X=X0,
        Y=Y0,
        Z=Z0,
        theta=theta0,
        phi=phi0,
        psi=psi0,
    )

    # Advancing element by element
    for ee, ll, aa, tt in zip(elements, drift_length, angle, tilt):

        # Store position and orientation at element entrance
        E_matrix.append(frame.E_matrix.copy())
        V.append(frame.XYZ.copy())

        _track_frame(
            frame, ee, length=ll, angle=aa, tilt=tt,
            backtrack=backtrack)

    # Last marker
    E_matrix.append(frame.E_matrix.copy())
    V.append(frame.XYZ.copy())

    # Return data for SurveyTable object
    return V, E_matrix


def _track_frame(p, element, *, length, angle, tilt, backtrack=False):
    if hasattr(element, 'track_frame'):
        element.track_frame(p, backtrack=backtrack)
    else:
        if backtrack:
            length = -length
            angle = -angle
        p.arc(length=length, angle=angle, tilt=tilt)
    return p


def track_frame(p, element, backtrack=False):
    """Track a frame through one beam element, mutating it in place.

    Elements exposing ``track_frame(frame, backtrack=False)`` are dispatched
    directly. For other elements, a temporary one-element line is used to
    obtain the same standard length, bend angle, and tilt employed by
    :meth:`Line.survey`.

    Parameters
    ----------
    p : Frame
        Frame to mutate.
    element : BeamElement
        Element through which to propagate the frame.
    backtrack : bool, optional
        Propagate backward through the element.

    Returns
    -------
    Frame
        The input frame ``p`` after in-place propagation.
    """
    if hasattr(element, 'track_frame'):
        return _track_frame(
            p, element, length=0, angle=0, tilt=0,
            backtrack=backtrack)

    # Local import avoids the survey/line import cycle.
    from ..line import Line

    if hasattr(element, '_parent') and hasattr(element, 'parent_name'):
        element_name = '_element'
        while element_name == element.parent_name:
            element_name = f'_{element_name}'
        elements = {
            element.parent_name: element._parent,
            element_name: element,
        }
        line = Line(elements=elements, element_names=[element_name])
    else:
        line = Line(elements=[element])

    table = line.get_table(attr=True)
    length = table.length[0] if table.isthick[0] else 0
    return _track_frame(
        p,
        element,
        length=length,
        angle=table.angle[0],
        tilt=table.rot_s_rad[0],
        backtrack=backtrack,
    )


def _get_survey_quantities_from_v_w(V, E_matrix):

    E_matrix = np.array(E_matrix)
    V = np.array(V)

    theta, phi, psi = _angles_from_E_matrix(E_matrix)

    ex = E_matrix[:, :, 0]
    ey = E_matrix[:, :, 1]
    ez = E_matrix[:, :, 2]
    X = V[:, 0]
    Y = V[:, 1]
    Z = V[:, 2]

    return {
        'X': X,
        'Y': Y,
        'Z': Z,
        'theta': np.unwrap(theta),
        'phi': np.unwrap(phi),
        'psi': np.unwrap(psi),
        'ex': ex,
        'ey': ey,
        'ez': ez,
        'XYZ': V.copy(),
        'E_matrix': E_matrix.copy(),
        'p0': V.copy(), # deprecated
        'W': E_matrix # deprecated
    }


def survey_relative_transform(survey: SurveyTable, source: str | int, destination: str | int, reversed=False) -> np.ndarray:
    """Generate a 3D transformation matrix from survey point `source` to `destination`.

    If `reversed`, take the transformation that points from the end point of `source` to the end point of `destination`.
    """

    if reversed:
        if source != survey.name[-1]:
            source = survey.rows.get_index(source) + 1
        if destination != survey.name[-1]:
            destination = survey.rows.get_index(destination) + 1
    else:
        if isinstance(source, str):
            source = survey.rows.get_index(source)
        if isinstance(destination, str):
            destination = survey.rows.get_index(destination)

    src_row = survey.rows[source]
    dest_row = survey.rows[destination]

    src_matrix = np.eye(4)
    src_matrix[:3, :3] = src_row.E_matrix
    src_matrix[:3, 3] = src_row.XYZ

    dest_matrix = np.eye(4)
    dest_matrix[:3, :3] = dest_row.E_matrix
    dest_matrix[:3, 3] = dest_row.XYZ

    return np.linalg.inv(src_matrix) @ dest_matrix
