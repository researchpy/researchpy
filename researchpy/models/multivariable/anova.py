# -*- coding: utf-8 -*-
"""
Analysis of Variance (ANOVA)

This module provides the Anova class for conducting analysis of variance
using sum of squares types I, II, and III.
"""

import re

import numpy as np
import scipy.stats
import patsy
import pandas as pd

from researchpy.models.linear_model import LinearModel
from researchpy.utility import rounder, patsy_term_cleaner


class Anova(LinearModel):
    """
    Analysis of Variance (ANOVA) for factorial designs.

    This class extends OLS to provide ANOVA functionality including:
    - Type I (Sequential) sum of squares
    - Type II (Hierarchical/Partially sequential) sum of squares
    - Type III (Marginal/Orthogonal) sum of squares
    - Partial effect size measures (Eta squared, Epsilon squared, Omega squared)

    Parameters
    ----------
    formula_like : str
        A string representing a valid Patsy formula (e.g., "y ~ C(factor1) + C(factor2)").
        See https://patsy.readthedocs.io/en/latest/ for formula syntax.
    data : dict or DataFrame, optional
        Data containing the variables referenced in the formula.
    sum_of_squares : int, optional
        Type of sum of squares to calculate. Options are:
        - 1 or "I": Sequential sum of squares
        - 2 or "II": Hierarchical/Partially sequential sum of squares
        - 3 or "III": Marginal/Orthogonal sum of squares (default)

    Attributes
    ----------
    factor_effects : dict
        Dictionary containing factor-level results including:
        - 'Source': Factor names
        - 'Sum of Squares': Sum of squares for each factor
        - 'Degrees of Freedom': Degrees of freedom for each factor
        - 'Mean Squares': Mean squares for each factor
        - 'F value': F-statistics for each factor
        - 'p-value': p-values for F-tests
        - 'Eta squared': Partial eta squared effect sizes
        - 'Epsilon squared': Partial epsilon squared effect sizes
        - 'Omega squared': Partial omega squared effect sizes

    Notes
    -----
    **Type I (Sequential) Sum of Squares:**
    Calculated by measuring the reduction in the model's overall sum of square
    error by subtracting the sum of square error for each term from the sum of
    square error from the model before.

    **Type II (Hierarchical) Sum of Squares:**
    The reduction in residual error due to adding the term to the model after
    all other terms except those that contain it.

    **Type III (Marginal) Sum of Squares:**
    Gives the sum of squares that would be obtained for each factor if it were
    to be entered into the model last. Best for models with interactions.

    Examples
    --------
    >>> import researchpy as rp
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'y': [1, 2, 3, 4, 5, 6],
    ...     'group': ['A', 'A', 'B', 'B', 'C', 'C']
    ... })
    >>> model = rp.Anova("y ~ C(group)", data=df)
    >>> model.results()

    See Also
    --------
    OLS : Ordinary Least Squares regression (parent class)
    """

    # ------------------------------------------------------------------ #
    #                        Helper Methods                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_scalar(val):
        """
        Safely convert a numpy array or scalar to a Python float.

        Handles 0-d arrays, 1-element arrays, and plain scalars uniformly
        without bare except blocks.

        Parameters
        ----------
        val : scalar, ndarray, or matrix
            Value to convert.

        Returns
        -------
        float
        """
        return float(np.asarray(val).item())

    @staticmethod
    def _compute_sse_from_design(x, y):
        """
        Compute the error sum of squares (SSE) given a design matrix and
        dependent variable vector.

        Computes: H = X(X'X)^{-1}X', ŷ = Hy, e = y - ŷ, SSE = e'e

        Parameters
        ----------
        x : ndarray
            Design matrix (n × p).
        y : ndarray
            Dependent variable vector (n × 1).

        Returns
        -------
        float
            The error (residual) sum of squares.
        """
        try:
            h = x @ np.linalg.inv(x.T @ x) @ x.T
        except np.linalg.LinAlgError:
            h = x @ np.linalg.pinv(x.T @ x) @ x.T

        predicted_y = h @ y
        residuals = y - predicted_y
        return residuals.T @ residuals

    def _compute_factor_stats(self, sum_of_square_factor, degrees_of_freedom_factor):
        """
        Compute mean square, F-statistic, p-value, and partial effect size
        measures for a single factor.

        All computations use the model-level MSE and SS values stored in
        ``self.model_data``.

        Parameters
        ----------
        sum_of_square_factor : float or ndarray
            Sum of squares attributable to this factor.
        degrees_of_freedom_factor : int or float
            Degrees of freedom for this factor.

        Returns
        -------
        dict
            Keys: 'msr_f', 'f_value', 'f_p_value', 'eta_sq', 'epsilon_sq', 'omega_sq'
        """
        mse = self.model_data["mse"]
        ss_residual = self.model_data["sum_of_square_residual"]
        ss_total = self.model_data["sum_of_square_total"]
        df_residual = self.model_data["degrees_of_freedom_residual"]

        # Mean Square - Factor
        msr_f = sum_of_square_factor * (1 / degrees_of_freedom_factor)

        # F-statistic and p-value
        f_value = msr_f / mse
        f_p_value = scipy.stats.f.sf(f_value, degrees_of_freedom_factor, df_residual)

        # Partial Effect Size Measures
        eta_sq = sum_of_square_factor / (sum_of_square_factor + ss_residual)

        epsilon_sq = (degrees_of_freedom_factor * (msr_f - mse)) / \
            (sum_of_square_factor + ss_total)

        omega_sq = (degrees_of_freedom_factor * (msr_f - mse)) / \
            ((degrees_of_freedom_factor * msr_f) + (self.n - degrees_of_freedom_factor) * mse)

        return {
            "msr_f": msr_f,
            "f_value": f_value,
            "f_p_value": f_p_value,
            "eta_sq": eta_sq,
            "epsilon_sq": epsilon_sq,
            "omega_sq": omega_sq,
        }

    @classmethod
    def _append_factor_effects(cls, factor_effects, source, ss_factor,
                               df_factor, stats):
        """
        Append a single factor's results to the factor_effects accumulator dict.

        Uses ``_to_scalar()`` for safe numeric conversion.

        Parameters
        ----------
        factor_effects : dict
            The accumulator dictionary being built up during ANOVA computation.
        source : str
            Term name for this factor.
        ss_factor : float or ndarray
            Sum of squares for this factor.
        df_factor : float or int
            Degrees of freedom for this factor.
        stats : dict
            Output from ``_compute_factor_stats()``.
        """
        factor_effects["Source"].append(source)
        factor_effects["Sum of Squares"].append(cls._to_scalar(ss_factor))
        factor_effects["Degrees of Freedom"].append(cls._to_scalar(df_factor))
        factor_effects["Mean Squares"].append(cls._to_scalar(stats["msr_f"]))
        factor_effects["F value"].append(cls._to_scalar(stats["f_value"]))
        factor_effects["p-value"].append(cls._to_scalar(stats["f_p_value"]))
        factor_effects["Eta squared"].append(cls._to_scalar(stats["eta_sq"]))
        factor_effects["Epsilon squared"].append(cls._to_scalar(stats["epsilon_sq"]))
        factor_effects["Omega squared"].append(cls._to_scalar(stats["omega_sq"]))

    @staticmethod
    def _build_type3_terms(term_names):
        """
        Convert patsy Treatment-coded term names to Sum-coded equivalents
        for Type III SS computation.

        Handles both main effects and interaction terms, preserving any
        user-specified reference categories by replacing Treatment(...) with Sum.

        Parameters
        ----------
        term_names : list of str
            Original term names from the design info (e.g., from
            ``self._IV_design_info.term_names``).

        Returns
        -------
        list of str
            Term names rewritten for Sum coding.
        """
        reference_pattern = re.compile(r'(?<=,|\s)(Treatment\(.*\))(?=\))')

        terms_sum = []
        for term in term_names:
            if "Treatment" in term:
                split_terms = term.split(":")

                if len(split_terms) == 1:
                    terms_sum.append(
                        re.sub(reference_pattern, 'Sum', split_terms[0]))
                else:
                    interaction_terms = []
                    for intterm in split_terms:
                        if "Treatment" in intterm:
                            interaction_terms.append(
                                re.sub(reference_pattern, 'Sum', intterm))
                        else:
                            interaction_terms.append(
                                intterm.replace(")", ", Sum)"))
                    terms_sum.append(':'.join(interaction_terms))
            else:
                terms_sum.append(term.replace(")", ", Sum)"))

        return terms_sum

    @staticmethod
    def _new_factor_effects_dict():
        """Return a fresh factor_effects accumulator dictionary."""
        return {
            "Source": [],
            "Sum of Squares": [],
            "Degrees of Freedom": [],
            "Mean Squares": [],
            "F value": [],
            "p-value": [],
            "Eta squared": [],
            "Epsilon squared": [],
            "Omega squared": [],
        }

    # ------------------------------------------------------------------ #
    #                        Constructor                                  #
    # ------------------------------------------------------------------ #

    def __init__(self, formula_like, data=None, sum_of_squares=3, conf_level=0.95, display_summary=True,
                 table_decimals=None):

        if data is None:
            data = {}

        self._test_stat_name = "t"
        self._CI_LEVEL = conf_level

        super().__init__(formula_like, data, conf_level=conf_level, display_summary=False,
                         table_decimals=table_decimals)

        self.__name__ = "Researchpy.ANOVA"

        # Validate sum_of_squares parameter
        valid_ss_types = [1, 2, 3, "I", "II", "III"]
        if sum_of_squares not in valid_ss_types:
            raise ValueError(
                f"sum_of_squares must be one of {valid_ss_types}, "
                f"got {sum_of_squares!r}"
            )

        # Store for potential re-use / inspection
        self._sum_of_squares_type = sum_of_squares

        # Compute factor-level ANOVA effects
        self.factor_effects = self._compute_factor_effects(data, sum_of_squares)

        # Display the model results summary
        if display_summary:
            self.summary()


    def _compute_factor_effects(self, data, sum_of_squares):
        """
        Compute per-factor ANOVA statistics based on the requested SS type.

        Parameters
        ----------
        data : DataFrame or dict
            The data used for building design matrices.
        sum_of_squares : int or str
            Type of sum of squares (1/"I", 2/"II", 3/"III").

        Returns
        -------
        dict
            The factor_effects dictionary with all per-factor statistics.
        """
        factor_effects = self._new_factor_effects_dict()

        if sum_of_squares in ["I", 1]:
            self._compute_ss_type1(factor_effects, data)

        elif sum_of_squares in ["II", 2]:
            self._compute_ss_type2(factor_effects, data)

        elif sum_of_squares in ["III", 3]:
            self._compute_ss_type3(factor_effects, data)

        return factor_effects


    def _compute_ss_type1(self, factor_effects, data):
        """
        Compute Type I (Sequential) sum of squares.

        Type I SS is calculated by measuring the reduction in the model's
        overall sum of square error by subtracting the SSE for each term
        from the SSE of the model before it. The difference is the factor's
        sum of squares.

        Parameters
        ----------
        factor_effects : dict
            Accumulator dictionary to append results into.
        data : DataFrame or dict
            Data for building design matrices.
        """
        previous_sse = self.model_data["sum_of_square_total"]
        terms_to_include = []

        for term in self._IV_design_info.term_names:
            if term.strip().upper() == "INTERCEPT":
                terms_to_include.append(term)
                continue

            if term not in terms_to_include:
                terms_to_include.append(term)

            # Build design matrix for terms included so far
            design_info = self._IV_design_info.subset(terms_to_include)
            x = np.asarray(patsy.build_design_matrices([design_info], data))[0]

            # Compute SSE for this cumulative model
            current_sse = self._compute_sse_from_design(x, self.DV)

            # Factor SS = reduction in SSE from adding this term
            ss_factor = previous_sse - current_sse

            # Degrees of freedom for this factor
            term_subset = self._IV_design_info.subset(term)
            term_design = np.asarray(
                patsy.build_design_matrices([term_subset], data))[0]
            df_factor = np.linalg.matrix_rank(term_design) - 1

            # Compute F, p-value, and effect sizes
            stats = self._compute_factor_stats(ss_factor, df_factor)

            # Store results
            self._append_factor_effects(
                factor_effects, term, ss_factor, df_factor, stats)

            # Update previous SSE for next iteration
            previous_sse = current_sse


    def _compute_ss_type2(self, factor_effects, data):
        """
        Compute Type II (Hierarchical/Partially Sequential) sum of squares.

        Type II SS is the reduction in residual error due to adding the term
        to the model after all other terms except those that contain it.

        Parameters
        ----------
        factor_effects : dict
            Accumulator dictionary to append results into.
        data : DataFrame or dict
            Data for building design matrices.
        """
        # Use complete cases for Type II (consistent with original implementation)
        complete_data = data[~data.isna().any(axis=1)]

        for current_term in self._IV_design_info.term_names:
            if current_term.strip().upper() == "INTERCEPT":
                continue

            # Build reduced model: all terms except those that contain current_term
            terms_in_model = list(self._IV_design_info.term_names)
            for term in self._IV_design_info.term_names:
                if term.strip().upper() == "INTERCEPT":
                    continue
                if current_term in term:
                    terms_in_model.remove(term)

            # Build full comparison model: reduced + current_term
            compare_model = list(terms_in_model)
            compare_model.append(current_term)

            # Build design matrices
            design_reduced = self._IV_design_info.subset(terms_in_model)
            x_reduced = np.asarray(patsy.build_design_matrices(
                [design_reduced], complete_data))[0]

            design_full = self._IV_design_info.subset(compare_model)
            x_full = np.asarray(patsy.build_design_matrices(
                [design_full], complete_data))[0]

            # Compute SSE for both models
            sse_reduced = self._compute_sse_from_design(x_reduced, self.DV)
            sse_full = self._compute_sse_from_design(x_full, self.DV)

            # Factor SS = SSE(reduced) - SSE(full)
            ss_factor = sse_reduced - sse_full

            # Degrees of freedom for this factor
            current_term_subset = self._IV_design_info.subset(current_term)
            current_term_design = np.asarray(
                patsy.build_design_matrices([current_term_subset], data))[0]
            df_factor = np.linalg.matrix_rank(current_term_design) - 1

            # Compute F, p-value, and effect sizes
            stats = self._compute_factor_stats(ss_factor, df_factor)

            # Store results
            self._append_factor_effects(
                factor_effects, current_term, ss_factor, df_factor, stats)


    def _compute_ss_type3(self, factor_effects, data):
        """
        Compute Type III (Marginal/Orthogonal) sum of squares.

        Type III SS gives the sum of squares that would be obtained for each
        factor if it were entered into the model last. This requires re-fitting
        the model using Sum coding (rather than Treatment/dummy coding) to
        ensure orthogonality.

        Best for models with interactions, where each main effect can be
        interpreted independently.

        Parameters
        ----------
        factor_effects : dict
            Accumulator dictionary to append results into.
        data : DataFrame or dict
            Data for building design matrices.
        """
        # Convert Treatment-coded terms to Sum-coded equivalents
        the_terms_3 = self._build_type3_terms(self._IV_design_info.term_names)

        # Re-fit the model using Sum coding
        full_model_formula = (
            self._DV_design_info.term_names[0]
            + " ~ "
            + " + ".join(the_terms_3[1:])
        )
        y, x_full = patsy.dmatrices(full_model_formula, data, eval_env=1)

        # For each term, compute SS by comparing full model to model-without-term
        for current_term in the_terms_3:
            if current_term.strip().upper() == "INTERCEPT":
                continue

            # Build the Sum-coded terms list for the reduced model (all terms minus current)
            terms_in_model = list(the_terms_3)
            terms_in_model.remove(current_term)

            # Build design matrix for reduced model
            design_reduced = x_full.design_info.subset(terms_in_model)
            x_reduced = np.asarray(patsy.build_design_matrices(
                [design_reduced], data))[0]

            # Compute SSE for reduced model
            sse_reduced = self._compute_sse_from_design(x_reduced, self.DV)

            # Factor SS = SSE(reduced) - SSE(full model)
            ss_factor = sse_reduced - self.model_data["sum_of_square_residual"]

            # Degrees of freedom for this factor
            current_term_subset = x_full.design_info.subset(current_term)
            current_term_design = np.asarray(
                patsy.build_design_matrices([current_term_subset], data))[0]
            df_factor = np.linalg.matrix_rank(current_term_design) - 1

            # Compute F, p-value, and effect sizes
            stats = self._compute_factor_stats(ss_factor, df_factor)

            # Store results
            self._append_factor_effects(
                factor_effects, current_term, ss_factor, df_factor, stats)


    # ...existing code...
    def results(self, return_type="Dataframe", decimals=4, pretty_format=True):
        """
        Return the ANOVA results as a descriptives + ANOVA table tuple.

        Parameters
        ----------
        return_type : str, optional
            ``"Dataframe"`` (default) or ``"Dictionary"``.
        decimals : int, optional
            Number of decimal places for rounding. Default is 4.
        pretty_format : bool, optional
            If True (default), empty-string placeholders are used for cells
            that don't apply (e.g., F-value for Residual row) — suitable for
            display. If False, ``np.nan`` is used instead — suitable for
            programmatic consumption.

        Returns
        -------
        tuple
            ``(descriptives, anova_table)`` where each element is a DataFrame
            (if ``return_type="Dataframe"``) or dict (if ``return_type="Dictionary"``).
        """
        # Use np.nan for missing cells when not pretty-formatting, '' otherwise
        blank = '' if pretty_format else np.nan

        # --- Descriptives (identical for both paths) ---
        descriptives = {
            "Number of obs = ": self.n,
            "Root MSE = ": round(self.model_data["root_mse"], decimals),
            "R-squared = ": round(self.model_data["r squared"], decimals),
            "Adj R-squared = ": round(self.model_data["r squared adj."], decimals),
        }

        # --- ANOVA table columns that always appear ---
        base_columns = ["Source", "Sum of Squares", "Degrees of Freedom",
                        "Mean Squares", "F value", "p-value"]
        effect_size_columns = ["Eta squared", "Epsilon squared", "Omega squared"]

        # --- Model row ---
        top_source = ["Model", ''] if pretty_format else ["Model"]
        top_row = {
            "Source": top_source,
            "Sum of Squares": [round(self.model_data["sum_of_square_model"], decimals)] + ([blank] if pretty_format else []),
            "Degrees of Freedom": [round(self.model_data["degrees_of_freedom_model"], decimals)] + ([blank] if pretty_format else []),
            "Mean Squares": [round(self.model_data["msr"], decimals)] + ([blank] if pretty_format else []),
            "F value": [round(self.model_data["f_value_model"], decimals)] + ([blank] if pretty_format else []),
            "p-value": [round(self.model_data["f_p_value_model"], decimals)] + ([blank] if pretty_format else []),
        }

        # Effect size columns only included when pretty_format (backward-compatible)
        if pretty_format:
            top_row["Eta squared"] = [round(self.model_data["Eta squared"], decimals), blank]
            top_row["Epsilon squared"] = [round(self.model_data["Epsilon squared"], decimals), blank]
            top_row["Omega squared"] = [round(self.model_data["Omega squared"], decimals), blank]

        # --- Factor rows ---
        factors = self.factor_effects.copy()
        factors["Source"] = [patsy_term_cleaner(term)
                             for term in self._IV_design_info.term_names[1:]]
        for col in base_columns[1:]:
            rounder(factors[col], decimals=decimals)
        if pretty_format:
            for col in effect_size_columns:
                rounder(factors[col], decimals=decimals)

        # --- Residual + Total rows ---
        bottom_source = ['', "Residual", "Total"] if pretty_format else ["Residual", "Total"]
        n_blank_prefix = 1 if pretty_format else 0
        bottom_row = {
            "Source": bottom_source,
            "Sum of Squares": [blank] * n_blank_prefix + [round(self.model_data["sum_of_square_residual"], decimals),
                                                           round(self.model_data["sum_of_square_total"], decimals)],
            "Degrees of Freedom": [blank] * n_blank_prefix + [round(self.model_data["degrees_of_freedom_residual"], decimals),
                                                               round(self.model_data["degrees_of_freedom_total"], decimals)],
            "Mean Squares": [blank] * n_blank_prefix + [round(self.model_data["mse"], decimals),
                                                         round(self.model_data["mst"], decimals)],
            "F value": [blank] * (n_blank_prefix + 2),
            "p-value": [blank] * (n_blank_prefix + 2),
        }

        if pretty_format:
            for col in effect_size_columns:
                bottom_row[col] = [blank] * (n_blank_prefix + 2)

        # --- Assemble the full ANOVA table ---
        all_columns = base_columns + (effect_size_columns if pretty_format else [])
        results_dict = {}
        for col in all_columns:
            results_dict[col] = top_row[col] + factors[col] + bottom_row[col]

        # --- Return ---
        if return_type == "Dataframe":
            return (pd.DataFrame.from_dict(descriptives, orient="index"),
                    pd.DataFrame.from_dict(results_dict))

        elif return_type == "Dictionary":
            return (descriptives, results_dict)

        else:
            raise ValueError(
                "Not a valid return type option, please use either "
                "'Dataframe' or 'Dictionary'."
            )


    def regression_table(self, return_type="Dataframe", pretty_format=True,
                         decimals={"Coef.": 2, "Std. Err.": 4, "test_stat": 4, "test_stat_p": 4, "CI": 2,
                                   "Root MSE": 4, "R-squared": 4, "Adj R-squared": 4, "Sum of Squares": 4,
                                   'Degrees of Freedom': 1, 'Mean Squares': 4, 'Effect size': 4},
                         *args):

        return super()._get_results(return_type=return_type, pretty_format=pretty_format, table_decimals=decimals)[2]


    def predict(self, estimate=None):
        return super().predict(estimate=estimate)


    # ------------------------------------------------------------------ #
    #                   summary() overrides for ANOVA                     #
    # ------------------------------------------------------------------ #
    def _get_summary_parts(self):
        """
        Return (model_description_df, coef_df) for the ANOVA summary.

        Calls ``self.results()`` with stdout suppressed (to avoid the
        "Note:" side-effect print) and returns the two DataFrames that
        ``CoreModel.summary()`` needs.

        Returns
        -------
        tuple of (model_description_df, coef_df)
        """
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            model_description_df, coef_df = self.results(
                return_type="Dataframe", pretty_format=True, decimals=4
            )

        model_summary_df = None
        return model_summary_df, model_description_df, coef_df


    # ------------------------------------------------------------------ #
    #               Header overrides for ANOVA                            #
    # ------------------------------------------------------------------ #
    def _summary_header_left(self, width=78, model_summary_df=None):
        """
        Left header for ANOVA: just the model display name.

        Overrides OLS's mini-ANOVA table because the full ANOVA table
        is already shown in the body section.
        """
        return [self._get_model_display_name()]


    def _summary_header_right(self, width=78, descriptives_df=None):
        """
        Right header for ANOVA: fit statistics from the descriptives DataFrame.

        Anova's descriptives_df is *index-oriented* (stat names as index,
        values in column 0), unlike OLS's transposed single-row format.
        """
        if descriptives_df is not None:
            desc_lines = descriptives_df.to_string(header=False).split("\n")
            return [desc_lines[0]] + self._splice_f_stat_lines() + desc_lines[1:]

        return super()._summary_header_right(width)


    # ------------------------------------------------------------------ #
    #                    ANOVA table (body section)                        #
    # ------------------------------------------------------------------ #
    def _summary_coef_table(self, df_results, width=78, table_decimals=None):
        """
        Build the ANOVA table as the summary body using
        ``DataFrame.to_string()``.

        Overrides the coefficient table that OLS / CoreModel would normally
        show.  Displays Model, individual factor rows, Residual, and Total
        with SS, df, MS, F, p-value, Eta², Epsilon², Omega².

        Parameters
        ----------
        df_results : DataFrame or None
            The ANOVA results DataFrame from ``self.results()``.
        width : int
            Total character width of the output.
        table_decimals : dict, optional
            Dictionary specifying decimal places for different statistics.

        Returns
        -------
        str
            Formatted ANOVA table string.
        """
        # ---- Resolve decimal places -------------------------------------
        base_decimals = {
            "Sum of Squares": 4, "Degrees of Freedom": 1, "Mean Squares": 4,
            "test_stat": 4, "test_stat_p": 4, "Effect size": 4,
        }
        if table_decimals is not None:
            base_decimals = base_decimals | table_decimals
        dec = base_decimals

        # ---- Prepare the DataFrame for display -------------------------
        table = df_results.copy()

        # Keep blank separator rows — they are intentional visual spacers
        # added by results() for a non-cluttered presentation.

        # Rename to shorter column headers
        table = table.rename(columns={
            "Sum of Squares":    "SS",
            "Degrees of Freedom": "df",
            "Mean Squares":      "MS",
            "F value":           "F",
            "p-value":           "p-value",
            "Eta squared":       "Eta^2",
            "Epsilon squared":   "Eps^2",
            "Omega squared":     "Omega^2",
        })

        # Replace empty strings with NaN so na_rep handles them uniformly
        table = table.replace("", float("nan"))

        # Coerce numeric columns from object dtype to float so that
        # to_string() formatters are applied consistently.
        numeric_cols = ["SS", "df", "MS", "F", "p-value",
                        "Eta^2", "Eps^2", "Omega^2"]
        for col in numeric_cols:
            table[col] = pd.to_numeric(table[col], errors="coerce")

        # ---- Per-column formatters (using shared static methods) -----------
        formatters = {
            "SS":      self._fmt_float(dec.get("Sum of Squares", 4)),
            "df":      self._fmt_int,
            "MS":      self._fmt_float(dec.get("Mean Squares", 4)),
            "F":       self._fmt_float(dec.get("test_stat", 4)),
            "p-value": self._fmt_float(dec.get("test_stat_p", 4)),
            "Eta^2":   self._fmt_float(dec.get("Effect size", 4)),
            "Eps^2":   self._fmt_float(dec.get("Effect size", 4)),
            "Omega^2": self._fmt_float(dec.get("Effect size", 4)),
        }

        # ---- Build the output string -----------------------------------
        sep = "-" * width

        table_str = table.to_string(
            index=False,
            na_rep="",
            formatters=formatters,
            justify="right",
        )

        lines = [
            sep,
            table_str,
            sep,
            "Note: Effect size values for factors are partial.",
        ]

        return "\n".join(lines)


# Convenience aliases for users who prefer different naming conventions
ANOVA = Anova