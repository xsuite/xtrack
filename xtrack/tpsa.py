import numpy as np
import math

class TPSA:
    def __init__(self, tpsa_dict: dict, num_variables: int = None):
        self.tpsa_dict = tpsa_dict
        self.monom_length = len(tpsa_dict)
        self.keys = list(tpsa_dict.keys())
        self.order = max(np.sum(tpsa_dict[k][0], axis=1).max() for k in tpsa_dict)
        if num_variables is None:
            self.num_variables = sum(1 for k in self.keys if len(self.tpsa_dict[k][1]) > 1)
        else:
            self.num_variables = num_variables

    def get_coeff(self, key: str, derived_var_arr: np.ndarray, verbose: str = False) -> float | list[float]:
        assert key in self.keys, f"Key {key} not in TPSA dict"

        if derived_var_arr.ndim == 1:
            coeff_index = np.where(np.all(derived_var_arr == self.tpsa_dict[key][0], axis=1))[0]
            if coeff_index.size == 0:
                if verbose:
                    print(f"WARNING: No coefficient found for key {key} with monomial {derived_var_arr} not found.")
                return 0.0
            elif coeff_index.size == 1:
                return self.tpsa_dict[key][1][coeff_index[0]]
        else:
            assert derived_var_arr.ndim == 2, "Only 1D or 2D arrays supported"
            coeff_indices = [np.where(np.all(i == self.tpsa_dict[key][0], axis=1))[0] for i in derived_var_arr]
            coeff_values = [self.tpsa_dict[key][1][i[0]] if i.size > 0 else 0.0 for i in coeff_indices]
            return coeff_values

    def get_taylor_expansion_key(self, key: str, delta: np.ndarray) -> float:
        """
        Evaluate the Taylor expansion for a given key at specified variable values.

        Parameters:
        -----------
        key: str
            The key in the TPSA dict to evaluate.
        delta: np.ndarray
            The difference between initial and perturbed coordinates to evaluate the Taylor expansion at.

        Returns:
        --------
        float
            The evaluated Taylor expansion value.
        """
        assert key in self.keys, f"Key {key} not in TPSA dict"
        assert delta.ndim == 1, "delta must be a 1D array"
        assert delta.size == self.monom_length, f"delta must have size {self.monom_length}"

        total = 0.0
        tpsas_key = self.tpsa_dict[key]
        for i in range(tpsas_key[0].shape[0]):
            monom = tpsas_key[0][i]
            coeff = tpsas_key[1][i]
            order = int(np.sum(monom))
            term = 1/math.factorial(order) * coeff
            for j in range(self.monom_length):
                if monom[j] != 0:
                    term *= (delta[j]) ** monom[j]
            total += term
        return total

    def get_taylor_expansion_all(self, delta: np.ndarray) -> np.ndarray:
        """
        Evaluate the Taylor expansion for all keys at specified variable values.

        Parameters:
        -----------
        var_values: np.ndarray
            A 1D array of variable values at which to evaluate the Taylor expansion.
        init_values: np.ndarray
            A 1D array of initial variable values (the expansion point).

        Returns:
        --------
        dict
            A dictionary with keys as in the TPSA dict and values as the evaluated Taylor expansion values.
        """
        assert delta.ndim == 1, "delta must be a 1D array"
        assert delta.size == self.monom_length, f"delta must have size {self.monom_length}"

        return np.array([self.get_taylor_expansion_key(k, delta) for k in self.keys[:self.num_variables]])

    def calc_beta(self, dim: str) -> float:
        """
        Calculate the beta function from the TPSA representation.

        Parameters:
        -----------
        dim: str
            Dimension to calculate beta for ('x' or 'y').

        Returns:
        --------
        float
            The calculated beta function value.
        """
        assert dim in ['x', 'y'], "Dimension must be 'x' or 'y'"

        dim_ind = 2 if dim == 'y' else 0

        x_x_coeff = self.get_coeff(dim, _arr_from_pos([dim_ind], self.monom_length))
        x_px_coeff = self.get_coeff(dim, _arr_from_pos([dim_ind + 1], self.monom_length))

        return x_x_coeff**2 + x_px_coeff**2

    def calc_alpha(self, dim: str) -> float:
        """
        Calculate the alpha function from the TPSA representation.

        Parameters:
        -----------
        dim: str
            Dimension to calculate alpha for ('x' or 'y').

        Returns:
        --------
        float
            The calculated alpha function value.
        """
        assert dim in ['x', 'y'], "Dimension must be 'x' or 'y'"

        dim_ind = 2 if dim == 'y' else 0
        pdim = 'p' + dim

        x_x_coeff = self.get_coeff(dim, _arr_from_pos([dim_ind], self.monom_length))
        px_x_coeff = self.get_coeff(pdim, _arr_from_pos([dim_ind], self.monom_length))
        x_px_coeff = self.get_coeff(dim, _arr_from_pos([dim_ind + 1], self.monom_length))
        px_px_coeff = self.get_coeff(pdim, _arr_from_pos([dim_ind + 1], self.monom_length))

        return - x_x_coeff * px_x_coeff - x_px_coeff * px_px_coeff

    def calc_dispersion(self, dim: str) -> float:
        """
        Calculate the dispersion from the TPSA representation.

        Parameters:
        -----------
        dim: str
            Dimension to calculate dispersion for ('x', 'px', 'y' or 'py').

        Returns:
        --------
        float
            The calculated dispersion value.
        """
        assert dim in ['x', 'px', 'y', 'py'], "Dimension must be 'x', 'px', 'y' or 'py'"

        x_delta_coeff = self.get_coeff(dim, _arr_from_pos([5], self.monom_length))
        x_zeta_coeff = self.get_coeff(dim, _arr_from_pos([4], self.monom_length))
        zeta_delta_coeff = self.get_coeff('t', _arr_from_pos([5], self.monom_length))
        zeta_zeta_coeff = self.get_coeff('t', _arr_from_pos([4], self.monom_length))
        delta_delta_coeff = self.get_coeff('pt', _arr_from_pos([5], self.monom_length))
        delta_zeta_coeff = self.get_coeff('pt', _arr_from_pos([4], self.monom_length))

        m = x_delta_coeff - (x_zeta_coeff * zeta_delta_coeff) / zeta_zeta_coeff
        n = delta_delta_coeff - (delta_zeta_coeff * zeta_delta_coeff) / zeta_zeta_coeff

        return m / n

    def calc_beta_deriv(self, dim: str, var_index: int) -> float:
        """
        Calculate the derivative of the beta function with respect to a variable.

        Parameters:
        -----------
        dim: str
            Dimension to calculate beta derivative for ('x' or 'y').
        var_index: int
            Index of the variable to differentiate with respect to.

        Returns:
        --------
        float
            The calculated derivative of the beta function.
        """

        assert dim in ['x', 'y'], "Dimension must be 'x' or 'y'"
        assert self.order >= 2, "TPSA order must be at least 2 to calculate beta derivative"

        dim_ind = 2 if dim == 'y' else 0

        x_x_coeff = self.get_coeff(dim, _arr_from_pos([dim_ind], self.monom_length))
        x_px_coeff = self.get_coeff(dim, _arr_from_pos([dim_ind + 1], self.monom_length))

        x_x_var_coeff = self.get_coeff(dim, _arr_from_pos([dim_ind, var_index], self.monom_length))
        x_px_var_coeff = self.get_coeff(dim, _arr_from_pos([dim_ind + 1, var_index], self.monom_length))

        return 2 * (x_x_coeff * x_x_var_coeff + x_px_coeff * x_px_var_coeff)

    def calc_alpha_deriv(self, dim: str, var_index: int) -> float:
        """
        Calculate the derivative of the alpha function with respect to a variable.

        Parameters:
        -----------
        dim: str
            Dimension to calculate alpha derivative for ('x' or 'y').
        var_index: int
            Index of the variable to differentiate with respect to.

        Returns:
        --------
        float
            The calculated derivative of the alpha function.
        """

        assert dim in ['x', 'y'], "Dimension must be 'x' or 'y'"
        assert self.order >= 2, "TPSA order must be at least 2 to calculate alpha derivative"

        dim_ind = 2 if dim == 'y' else 0
        pdim = 'p' + dim

        x_x_coeff = self.get_coeff(dim, _arr_from_pos([dim_ind], self.monom_length))
        px_x_coeff = self.get_coeff(pdim, _arr_from_pos([dim_ind], self.monom_length))
        x_x_var_coeff = self.get_coeff(dim, _arr_from_pos([dim_ind, var_index], self.monom_length))
        px_x_var_coeff = self.get_coeff(pdim, _arr_from_pos([dim_ind, var_index], self.monom_length))

        x_px_coeff = self.get_coeff(dim, _arr_from_pos([dim_ind + 1], self.monom_length))
        px_px_coeff = self.get_coeff(pdim, _arr_from_pos([dim_ind + 1], self.monom_length))
        x_px_var_coeff = self.get_coeff(dim, _arr_from_pos([dim_ind + 1, var_index], self.monom_length))
        px_px_var_coeff = self.get_coeff(pdim, _arr_from_pos([dim_ind + 1, var_index], self.monom_length))

        return - (x_x_var_coeff * px_x_coeff + x_x_coeff * px_x_var_coeff
                  + x_px_var_coeff * px_px_coeff + x_px_coeff * px_px_var_coeff)

    def calc_dispersion_deriv(self, dim: str, var_index: int) -> float:
        """
        Calculate the derivative of the dispersion with respect to a variable.

        Parameters:
        -----------
        dim: str
            Dimension to calculate dispersion derivative for ('x', 'px', 'y' or 'py').
        var_index: int
            Index of the variable to differentiate with respect to.

        Returns:
        --------
        float
            The calculated derivative of the dispersion.
        """

        assert dim in ['x', 'px', 'y', 'py'], "Dimension must be 'x', 'px', 'y' or 'py'"
        assert self.order >= 2, "TPSA order must be at least 2 to calculate dispersion derivative"

        x_delta_coeff = self.get_coeff(dim, _arr_from_pos([5], self.monom_length))
        x_zeta_coeff = self.get_coeff(dim, _arr_from_pos([4], self.monom_length))
        zeta_delta_coeff = self.get_coeff('t', _arr_from_pos([5], self.monom_length))
        zeta_zeta_coeff = self.get_coeff('t', _arr_from_pos([4], self.monom_length))
        delta_delta_coeff = self.get_coeff('pt', _arr_from_pos([5], self.monom_length))
        delta_zeta_coeff = self.get_coeff('pt', _arr_from_pos([4], self.monom_length))

        x_delta_var_coeff = self.get_coeff(dim, _arr_from_pos([5, var_index], self.monom_length))
        x_zeta_var_coeff = self.get_coeff(dim, _arr_from_pos([4, var_index], self.monom_length))
        zeta_delta_var_coeff = self.get_coeff('t', _arr_from_pos([5, var_index], self.monom_length))
        delta_delta_var_coeff = self.get_coeff('pt', _arr_from_pos([5, var_index], self.monom_length))
        delta_zeta_var_coeff = self.get_coeff('pt', _arr_from_pos([4, var_index], self.monom_length))

        m = x_delta_coeff - (x_zeta_coeff * zeta_delta_coeff) / zeta_zeta_coeff
        n = delta_delta_coeff - (delta_zeta_coeff * zeta_delta_coeff) / zeta_zeta_coeff

        dm = x_delta_var_coeff - (x_zeta_var_coeff * zeta_delta_coeff - x_zeta_coeff * zeta_delta_var_coeff) / zeta_zeta_coeff**2

        dn = delta_delta_var_coeff - (delta_zeta_var_coeff * zeta_delta_coeff - delta_zeta_coeff * zeta_delta_var_coeff) / zeta_zeta_coeff**2

        return (dm * n - m * dn) / n**2

    def calc_phase_advance_deriv(self, dim: str, var_index: int) -> float:
        """
        Calculate the derivative of the phase advance with respect to a variable.

        Parameters:
        -----------
        dim: str
            Dimension to calculate phase advance derivative for ('x' or 'y').
        var_index: int
            Index of the variable to differentiate with respect to.

        Returns:
        --------
        float
            The calculated derivative of the phase advance.
        """

        assert dim in ['x', 'y'], "Dimension must be 'x' or 'y'"
        #assert self.order >= 2, "TPSA order must be at least 2 to calculate phase advance derivative"
        dim_ind = 2 if dim == 'y' else 0

        x_x_coeff = self.get_coeff(dim, _arr_from_pos([dim_ind], self.monom_length))
        x_px_coeff = self.get_coeff(dim, _arr_from_pos([dim_ind + 1], self.monom_length))
        x_x_var_coeff = self.get_coeff(dim, _arr_from_pos([dim_ind, var_index], self.monom_length))
        x_px_var_coeff = self.get_coeff(dim, _arr_from_pos([dim_ind + 1, var_index], self.monom_length))

        return (x_x_coeff/(x_x_coeff**2 + x_px_coeff**2) * x_px_var_coeff - x_px_coeff/(x_x_coeff**2 + x_px_coeff**2) * x_x_var_coeff) / (2*math.pi)

def _arr_from_pos(pos, length):
        return np.isin(np.arange(length), pos).astype(int)
