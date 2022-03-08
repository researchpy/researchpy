# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 08:07:51 2021

@author: Corey
"""

# Used
import numpy
import scipy.stats
import patsy
import pandas

from .summary import summarize
from .model import model
from .utility import *




class anova(model):

    def __init__(self, formula_like, data = {}, sum_of_squares = 3):
        super().__init__(formula_like, data, matrix_type = 1)

        self.grand_mean = numpy.sum(self.DV, axis = 0)

        self.model_data = {}


        ###########

        # j matrix of ones based on y
        self.model_data["j"] = numpy.ones((self.nobs, self.nobs))

        # identity matrix (i) based on x
        self.model_data["i"] = numpy.identity(self.nobs)

        # Hat matrix
        self.model_data["h"] = self.IV @ numpy.linalg.inv(self.IV.T @ self.IV) @ self.IV.T

        # Estimation of betas
        self.model_data["betas"] = numpy.linalg.inv((self.IV.T @ self.IV)) @ self.IV.T @ self.DV

        # Predicted y values
        self.model_data["predicted_y"] = self.IV @ self.model_data["betas"]

        # Calculation of residuals (error)
        self.model_data["residuals"] = self.DV - self.model_data["predicted_y"]



        ###  Sum of Squares
        # Total sum of squares (SSTO)
        self.model_data["sum_of_square_total"] = float(self.DV.T @ self.DV - (1/self.nobs) * self.DV.T @ self.model_data["j"] @ self.DV)

        # Model sum of squares (SSR)
        self.model_data["sum_of_square_model"] = float(self.model_data["betas"].T @ self.IV.T @ self.DV - (1/self.nobs) * self.DV.T @ self.model_data["j"] @ self.DV)

        # Error sum of squares (SSE)
        self.model_data["sum_of_square_residual"] = float(self.model_data["residuals"].T @ self.model_data["residuals"])



        ### Degrees of freedom
        # Model
        self.model_data["degrees_of_freedom_model"] = numpy.linalg.matrix_rank(self.IV) - 1

        # Error
        self.model_data["degrees_of_freedom_residual"] = self.nobs - numpy.linalg.matrix_rank(self.IV)

        # Total
        self.model_data["degrees_of_freedom_total"] = self.nobs - 1



        ### Mean Square
        # Model (MSR)
        self.model_data["msr"] = self.model_data["sum_of_square_model"] * (1/self.model_data["degrees_of_freedom_model"])

        # Residual (error; MSE)
        self.model_data["mse"] = self.model_data["sum_of_square_residual"] * (1/self.model_data["degrees_of_freedom_residual"])

        #Total (MST)
        self.model_data["mst"] = self.model_data["sum_of_square_total"] * (1/self.model_data["degrees_of_freedom_total"])



        ## Root Mean Square Error
        self.model_data["root_mse"] = float(numpy.sqrt(self.model_data["mse"]))



        ### F-values
        # Model
        self.model_data["f_value_model"] = float(self.model_data["msr"] / self.model_data["mse"])
        self.model_data["f_p_value_model"] = scipy.stats.f.sf(self.model_data["f_value_model"], self.model_data["degrees_of_freedom_model"], self.model_data["degrees_of_freedom_residual"])


        ### R^2 Values
        # Model
        self.model_data["r squared"] = (self.model_data["sum_of_square_model"] / self.model_data["sum_of_square_total"])
        self.model_data["r squared adj."] = 1 - (self.model_data["degrees_of_freedom_total"]  / self.model_data["degrees_of_freedom_residual"]) * (self.model_data["sum_of_square_residual"] / self.model_data["sum_of_square_total"])
        self.model_data["Eta squared"] = self.model_data["r squared"]
        self.model_data["Omega squared"] = (self.model_data["degrees_of_freedom_model"] * (self.model_data["msr"] - self.model_data["mse"])) / (self.model_data["sum_of_square_total"] + self.model_data["mse"])




        ###########

        factor_effects = {"Source" : [],
                          "Sum of Squares" : [],
                          "Degrees of Freedom" : [],
                          "Mean Squares" : [],
                          "F value" : [],
                          "p-value" : [],
                          "Eta squared" : [],
                          "Omega squared" : []}



        ###########################################################
        #               CODUCTING THE TEST                        #
        ###########################################################
        if sum_of_squares in ["I", 1]:

            """
            Type I sum of squares is also known as sequential sum of squares and is calculated by
                measuring the reduction in the model's overall sum of square error by subtracting
                the sum of square error for each term from the sum of square error from the model
                before. The difference between the two is the sum of square factor effect for the
                current term.
            """

            previous_sum_of_squares_error = self.model_data["sum_of_square_total"]
            terms_to_include = []

            for term in self._IV_design_info.term_names:
                if term.strip().upper() == "INTERCEPT":
                    terms_to_include.append(term)
                    continue


                else:
                    if term not in terms_to_include:
                        terms_to_include.append(term)

                    # Getting design matrix set up
                    design_info2 = self._IV_design_info.subset(terms_to_include)
                    x = numpy.asarray(patsy.build_design_matrices([design_info2], data))[0]

                    # Hat matrix
                    h = x @ numpy.linalg.inv(x.T @ x) @ x.T

                    # Predicted y values
                    predicted_y = h @ self.DV

                    # Calculation of residuals (error)
                    residuals = self.DV - predicted_y

                    ### Error sum of squares (SSE)
                    sum_of_square_residual = residuals.T @ residuals

                    # Calculating the Factor Effect Sum of Squares(SSR, and SSE)
                    sum_of_square_factor = previous_sum_of_squares_error - sum_of_square_residual

                    ### Degrees of freedom - Factor
                    term_subset = self._IV_design_info.subset(term)
                    term_design = numpy.asarray(patsy.build_design_matrices([term_subset], data))[0]

                    degrees_of_freedom_factor = numpy.linalg.matrix_rank(term_design) - 1

                    ### Mean Square - Factor (MSR-F)
                    msr_f = sum_of_square_factor * (1 / degrees_of_freedom_factor)



                    ### F-values
                    # Model
                    f_value_model = msr_f / self.model_data["mse"]
                    f_p_value_model = scipy.stats.f.sf(f_value_model, degrees_of_freedom_factor, self.model_data["degrees_of_freedom_residual"])

                    ### Effect Size Measure(s)
                    eta_squared = sum_of_square_factor / (sum_of_square_factor + self.model_data["sum_of_square_residual"])
                    omega_squared = (sum_of_square_factor - (degrees_of_freedom_factor * self.model_data["mse"])) / (sum_of_square_factor + (self.nobs - degrees_of_freedom_factor) * self.model_data["mse"])





                    # Updating items
                    factor_effects["Source"].append(term)
                    factor_effects["Sum of Squares"].append(float(sum_of_square_factor))
                    factor_effects["Degrees of Freedom"].append(float(degrees_of_freedom_factor))
                    factor_effects["Mean Squares"].append(float(msr_f))
                    factor_effects["F value"].append(float(f_value_model))
                    factor_effects["p-value"].append(float(f_p_value_model))

                    factor_effects["Eta squared"].append(float(eta_squared))
                    factor_effects["omega squared"].append(float(omega_squared))
                    #factor_effects["r squared"].append("")
                    #factor_effects["r squared adj."].append("")

                    # Setting new Sum of Square Residual
                    previous_sum_of_squares_error = sum_of_square_residual

                self.factor_effects = factor_effects


        elif sum_of_squares in ["II", 2]:
            """
            Type II sum of square is also known as hierarchical or
                partially sequential. This type of sum of square is
                the reduction in residual error due to adding the term
                to the model after all other terms except those
                that contain it.
            """

            for current_term in self._IV_design_info.term_names:
                terms_in_model = self._IV_design_info.term_names

                if current_term.strip().upper() == "INTERCEPT":
                    continue

                else:
                    for term in self._IV_design_info.term_names:
                        if term.strip().upper() == "INTERCEPT":
                            continue

                        elif current_term in term:
                            terms_in_model.remove(term)

                        else:
                            continue

                    if current_term.strip().upper() == "INTERCEPT":
                        continue

                    else:
                        compare_model = [item for item in terms_in_model]
                        compare_model.append(current_term)



                # Getting design matrix set up
                design_info2 = self._IV_design_info.subset(terms_in_model)
                x = numpy.asarray(patsy.build_design_matrices([design_info2], data))[0]

                design_info3 = self._IV_design_info.subset(compare_model)
                x_c = numpy.asarray(patsy.build_design_matrices([design_info3], data))[0]

                # Hat matrix
                h = x @ numpy.linalg.inv(x.T @ x) @ x.T
                h_c = x_c @ numpy.linalg.inv(x_c.T @ x_c) @ x_c.T

                # Predicted y values
                predicted_y = h @ self.DV
                predicted_y_c = h_c @ self.DV

                # Calculation of residuals (error)
                residuals = self.DV - predicted_y
                residuals_c = self.DV - predicted_y_c

                ###  Error sum of squares (SSE)
                sum_of_square_residual = residuals.T @ residuals
                sum_of_square_residual_c = residuals_c.T @ residuals_c

                # Calculating the Factor Effect
                sum_of_square_factor = sum_of_square_residual -  sum_of_square_residual_c

                ### Degrees of freedom - Factor
                current_term_subset = self._IV_design_info.subset(current_term)
                current_term_design = numpy.asarray(patsy.build_design_matrices([current_term_subset], data))[0]

                degrees_of_freedom_factor = numpy.linalg.matrix_rank(current_term_design) - 1

                ### Mean Square - Factor (MSR-F)
                msr_f = sum_of_square_factor * (1 / degrees_of_freedom_factor)



                ### F-values
                # Model
                f_value_model = msr_f / self.model_data["mse"]
                f_p_value_model = scipy.stats.f.sf(f_value_model, degrees_of_freedom_factor, self.model_data["degrees_of_freedom_residual"])


                ### Effect Size Measure(s)
                eta_squared = sum_of_square_factor / (sum_of_square_factor + self.model_data["sum_of_square_residual"])
                omega_squared = (sum_of_square_factor - (degrees_of_freedom_factor * self.model_data["mse"])) / (sum_of_square_factor + (self.nobs - degrees_of_freedom_factor) * self.model_data["mse"])



                # Updating items
                factor_effects["Source"].append(current_term)
                factor_effects["Sum of Squares"].append(float(sum_of_square_factor))
                factor_effects["Degrees of Freedom"].append(float(degrees_of_freedom_factor))
                factor_effects["Mean Squares"].append(float(msr_f))
                factor_effects["F value"].append(float(f_value_model))
                factor_effects["p-value"].append(float(f_p_value_model))

                factor_effects["Eta squared"].append(float(eta_squared))
                factor_effects["Omega squared"].append(float(omega_squared))
                #factor_effects["r squared"].append("")
                #factor_effects["r squared adj."].append("")

            self.factor_effects = factor_effects



        elif sum_of_squares in ["III", 3]:

            """
            Type III sum of square is also known as marginal or orthogonal. This
                calculation of sum of squares give the sum of squares that would be
                obtained for each factor if it were to be entered into the model last.

            Best for if using an interaction (each main effect is able to be interpereted
                although not worthwhile if the interaction is significant)
            """


            reference_pattern = re.compile(r'(?<=,|\s)(Treatment\(.*\))(?=\))')

            the_terms_3 = []
            for term in self._IV_design_info.term_names:

                if "Treatment" in term:
                    split_terms = term.split(":")

                    if len(split_terms) == 1:
                        the_terms_3.append(re.sub(reference_pattern, 'Sum', split_terms[0]))

                    else:
                        interaction_terms = []

                        for intterm in split_terms:
                            if "Treatment" in intterm:
                                interaction_terms.append(re.sub(reference_pattern, 'Sum', intterm))
                            else:
                                interaction_terms.append(intterm.replace(")", ", Sum)"))

                        the_terms_3.append(':'.join(interaction_terms))

                else:
                    the_terms_3.append(term.replace(")", ", Sum)"))



            full_model = self._DV_design_info.term_names[0] + " ~ " + " + ".join(the_terms_3[1:])
            y, x_full = patsy.dmatrices(full_model, data, eval_env=1)


            for current_term in the_terms_3:

                terms_in_model = []
                for term in self._IV_design_info.term_names:
                    if "Treatment" in term:
                        split_terms = term.split(":")

                        if len(split_terms) == 1:
                            terms_in_model.append(re.sub(reference_pattern, 'Sum', split_terms[0]))

                        else:
                            interaction_terms = []

                            for intterm in split_terms:
                                if "Treatment" in intterm:
                                    interaction_terms.append(re.sub(reference_pattern, 'Sum', intterm))

                                else:
                                    interaction_terms.append(intterm.replace(")", ", Sum)"))

                            terms_in_model.append(':'.join(interaction_terms))

                    else:
                        terms_in_model.append(term.replace(")", ", Sum)"))



                compare_model = self._IV_design_info.term_names

                if current_term.strip().upper() == "INTERCEPT":
                    continue
                else:
                    terms_in_model.remove(current_term)



                # Getting design matrix set up
                design_info2 = x_full.design_info.subset(terms_in_model)
                x = numpy.asarray(patsy.build_design_matrices([design_info2], data))[0]

                # Hat matrix
                h = x @ numpy.linalg.inv(x.T @ x) @ x.T

                # Predicted y values
                predicted_y = h @ self.DV

                # Calculation of residuals (error)
                residuals = self.DV - predicted_y

                ###  Error sum of squares (SSE)
                sum_of_square_residual = residuals.T @ residuals

                # Calculating the Factor Effect
                sum_of_square_factor = sum_of_square_residual -  self.model_data["sum_of_square_residual"]

                ### Degrees of freedom - Factor
                current_term_subset = x_full.design_info.subset(current_term)
                current_term_design = numpy.asarray(patsy.build_design_matrices([current_term_subset], data))[0]

                degrees_of_freedom_factor = numpy.linalg.matrix_rank(current_term_design) - 1

                ### Mean Square - Factor (MSR-F)
                msr_f = sum_of_square_factor * (1 / degrees_of_freedom_factor)



                ### F-values
                # Model
                f_value_model = msr_f / self.model_data["mse"]
                f_p_value_model = scipy.stats.f.sf(f_value_model, degrees_of_freedom_factor, self.model_data["degrees_of_freedom_residual"])


                ### Effect Size Measure(s)
                eta_squared = sum_of_square_factor / (sum_of_square_factor + self.model_data["sum_of_square_residual"])
                omega_squared = (sum_of_square_factor - (degrees_of_freedom_factor * self.model_data["mse"])) / (sum_of_square_factor + (self.nobs - degrees_of_freedom_factor) * self.model_data["mse"])


                # Updating items
                factor_effects["Source"].append(current_term)
                factor_effects["Sum of Squares"].append(float(sum_of_square_factor))
                factor_effects["Degrees of Freedom"].append(float(degrees_of_freedom_factor))
                factor_effects["Mean Squares"].append(float(msr_f))
                factor_effects["F value"].append(float(f_value_model))
                factor_effects["p-value"].append(float(f_p_value_model))


                factor_effects["Eta squared"].append(float(eta_squared))
                factor_effects["Omega squared"].append(float(omega_squared))
                #factor_effects["r squared"].append("")
                #factor_effects["r squared adj."].append("")


                self.factor_effects = factor_effects


    def results(self, return_type = "Dataframe", decimals = 4, pretty_format = True):


        if pretty_format == True:

            descriptives = {

                    "Number of obs = " : self.nobs,
                    "Root MSE = " : round(self.model_data["root_mse"], decimals),
                    "R-squared = " : round(self.model_data["r squared"], decimals),
                    "Adj R-squared = " : round(self.model_data["r squared adj."], decimals)

                }

            top = {

                    "Source" : ["Model", ''],
                    "Sum of Squares" : [round(self.model_data["sum_of_square_model"], decimals), ''],
                    "Degrees of Freedom" : [round(self.model_data["degrees_of_freedom_model"], decimals), ''],
                    "Mean Squares" : [round(self.model_data["msr"], decimals), ''],
                    "F value" : [round(self.model_data["f_value_model"], decimals), ''],
                    "p-value" : [round(self.model_data["f_p_value_model"], decimals), ''],
                    "Eta squared" : [round(self.model_data["Eta squared"], decimals), ''],
                    "Omega squared" : [round(self.model_data["Omega squared"], decimals), '']

                    }

            factors = self.factor_effects.copy()
            factors["Source"] = [patsy_term_cleaner(term) for term in self._IV_design_info.term_names[1:]]
            rounder(factors["Sum of Squares"], decimals = decimals)
            rounder(factors["Degrees of Freedom"], decimals = decimals)
            rounder(factors["Mean Squares"], decimals = decimals)
            rounder(factors["F value"], decimals = decimals)
            rounder(factors["p-value"], decimals = decimals)
            rounder(factors["Eta squared"], decimals = decimals)
            rounder(factors["Omega squared"], decimals = decimals)

            bottom = {

                    "Source" : ['', "Residual", "Total"],
                    "Sum of Squares" : ['', round(self.model_data["sum_of_square_residual"], decimals), round(self.model_data["sum_of_square_total"], decimals)],
                    "Degrees of Freedom" : ['', round(self.model_data["degrees_of_freedom_residual"], decimals), round(self.model_data["degrees_of_freedom_total"], decimals)],
                    "Mean Squares" : ['', round(self.model_data["mse"], decimals), round(self.model_data["mst"], decimals)],
                    "F value" : ['', '', ''],
                    "p-value" : ['', '', ''],
                    "Eta squared" : ['', '', ''],
                    "Omega squared" : ['', '', '']

                    }


            results = {

                    "Source" : top["Source"] + factors["Source"] + bottom["Source"],
                    "Sum of Squares" : top["Sum of Squares"] + factors["Sum of Squares"]  + bottom["Sum of Squares"],
                    "Degrees of Freedom" : top["Degrees of Freedom"] + factors["Degrees of Freedom"] + bottom["Degrees of Freedom"],
                    "Mean Squares" : top["Mean Squares"] + factors["Mean Squares"] + bottom["Mean Squares"],
                    "F value" : top["F value"] + factors["F value"] + bottom["F value"],
                    "p-value" : top["p-value"] + factors["p-value"] + bottom["p-value"],
                    "Eta squared" : top["Eta squared"] + factors["Eta squared"] + bottom["Eta squared"],
                    "Omega squared" : top["Omega squared"] + factors["Omega squared"] + bottom["Omega squared"]

                    }

        else:

            descriptives = {

                    "Number of obs = " : self.nobs,
                    "Root MSE = " : round(self.model_data["root_mse"], decimals),
                    "R-squared = " : round(self.model_data["r squared"], decimals),
                    "Adj R-squared = " : round(self.model_data["r squared adj."], decimals)

                }


            top = {

                    "Source" : ["Model"],
                    "Sum of Squares" : [round(self.model_data["sum_of_square_model"], decimals)],
                    "Degrees of Freedom" : [round(self.model_data["degrees_of_freedom_model"], decimals)],
                    "Mean Squares" : [round(self.model_data["msr"], decimals)],
                    "F value" : [round(self.model_data["f_value_model"], decimals)],
                    "p-value" : [round(self.model_data["f_p_value_model"], decimals)]

                    }

            factors = self.factor_effects.copy()
            patsy_term_cleaner(factors["Source"])
            rounder(factors["Sum of Squares"], decimals = decimals)
            rounder(factors["Degrees of Freedom"], decimals = decimals)
            rounder(factors["Mean Squares"], decimals = decimals)
            rounder(factors["F value"], decimals = decimals)
            rounder(factors["p-value"], decimals = decimals)

            bottom = {

                    "Source" : ["Residual", "Total"],
                    "Sum of Squares" : [round(self.model_data["sum_of_square_residual"], decimals), round(self.model_data["sum_of_square_total"], decimals)],
                    "Degrees of Freedom" : [round(self.model_data["degrees_of_freedom_residual"], decimals), round(self.model_data["degrees_of_freedom_total"], decimals)],
                    "Mean Squares" : [round(self.model_data["mse"], decimals), round(self.model_data["mst"], decimals)],
                    "F value" : [numpy.nan, numpy.nan],
                    "p-value" : [numpy.nan, numpy.nan]

                    }


            results = {

                    "Source" : top["Source"] + factors["Source"] + bottom["Source"],
                    "Sum of Squares" : top["Sum of Squares"] + factors["Sum of Squares"]  + bottom["Sum of Squares"],
                    "Degrees of Freedom" : top["Degrees of Freedom"] + factors["Degrees of Freedom"] + bottom["Degrees of Freedom"],
                    "Mean Squares" : top["Mean Squares"] + factors["Mean Squares"] + bottom["Mean Squares"],
                    "F value" : top["F value"] + factors["F value"] + bottom["F value"],
                    "p-value" : top["p-value"] + factors["p-value"] + bottom["p-value"]

                    }



        if return_type == "Dataframe":

            print("\n"*2, "Note: Effect size values for factors are partial.", "\n"*2)
            return (pandas.DataFrame.from_dict(descriptives, orient = "index"), pandas.DataFrame.from_dict(results))


        elif return_type == "Dictionary":

            print("\n"*2, "Note: Effect size values for factors are partial.", "\n"*2)
            return (descriptives, results)

        else:

            print("Not a valid return type option, please use either 'Dataframe' or 'Dictionary'.")




    def regression_table(self, return_type = "Dataframe", decimals = 4, pretty_format = True, conf_level = 0.95):

        ### Variance-covariance matrices
        # Non-robust - from Applied Linear Statistical Models, pg. 203
        self.variance_covariance_residual_matrix = numpy.matrix(self.model_data["mse"] * (self.model_data["i"] - self.model_data["h"]))


        self.variance_covariance_beta_matrix = numpy.matrix(self.model_data["mse"] * numpy.linalg.inv(self.IV.T @ self.IV))

        #if robust_standard_errors == False:
            #self.beta_variance_covariance_matrix = np.matrix(self.mse * np.linalg.inv(self.x.T @ self.x))
        #elif robust_standard_errors == True:
            ## Not implemented yet so it's same as non-robust
         #   self.beta_variance_covariance_matrix = np.matrix(self.mse * np.linalg.inv(self.x.T @ self.x))



        ### Standard Errors
        self.standard_errors = (numpy.array(numpy.sqrt(self.variance_covariance_beta_matrix.diagonal()))).T




        ### Confidence Intrvals
        self.conf_int_lower = []
        self.conf_int_upper = []

        for beta, se in zip(self.model_data["betas"], self.standard_errors):
            lower, upper = scipy.stats.t.interval(conf_level, self.model_data["degrees_of_freedom_residual"], loc= beta, scale= se)

            self.conf_int_lower.append(float(lower))
            self.conf_int_upper.append(float(upper))



        ### T-stastics
        self.t_stastics = self.model_data["betas"] * (1 / self.standard_errors)
        # Two-sided p-value
        self.t_p_values = numpy.array([float(scipy.stats.t.sf(numpy.abs(t), self.model_data["degrees_of_freedom_residual"]) * 2) for t in self.t_stastics])



        ## Creating variable table information
        self.regression_description_info = {

            self._DV_design_info.term_names[0] : ["Coef.", "Std. Err.", "t", "p-value", "95% Conf. Interval"],

            }


        regression_info = {self._DV_design_info.term_names[0] : [],
                                "Coef." : [],
                                "Std. Err." : [],
                                "t" : [],
                                "p-value": [],
                                f"{int(conf_level * 100)}% Conf. Interval": []}


        for column, beta, stderr, t, p, l_ci, u_ci in zip(self._IV_design_info.column_names, self.model_data["betas"], self.standard_errors, self.t_stastics, self.t_p_values, self.conf_int_lower, self.conf_int_upper):

            regression_info[self._DV_design_info.term_names[0]].append(column)
            regression_info["Coef."].append(round(beta[0], decimals))
            regression_info["Std. Err."].append(round(stderr[0], decimals))
            regression_info["t"].append(round(t[0], decimals))
            regression_info["p-value"].append(round(p, decimals))
            regression_info[f"{int(conf_level * 100)}% Conf. Interval"].append([round(l_ci, decimals), round(u_ci, decimals)])


        regression_info = base_table(self._patsy_factor_information, self._mapping, self._rp_factor_information, pandas.DataFrame.from_dict(regression_info))



        if return_type == "Dataframe":

            return regression_info

        elif return_type == "Dictionary":


            return regression_info

        else:

            print("Not a valid return type option, please use either 'Dataframe' or 'Dictionary'.")
