import numpy
import scipy.stats
import patsy
import re
import pandas
from .summary import summarize




class difference_test(object):

    """

    A method that conducts various difference tests and effect size measures which
    will be returned as a Pandas DataFrame (default) or a Python dictionar object. Two
    objects will be returned for all tests; first object is descriptive statistic
    information, and the second object is the statistical testing information as
    well as any effect sizes that were specified.

    Available difference tests:

        [] Independent sampes t-test;
            > equal_variances = True, independent_samples = True
        [] Paired samples t-test;
            > equal_variances = True, independent_samples = False
        [] Welch's t-test;
            > equal_variances = False, independent_samples = True
        [] Wilcoxon signed-rank test.
            > equal_variances = False, independent_samples = False

    Effect size measures are passed in the -conduct()- step; available effect
    size options are: "Cohen's D", "Hedge's G", "Glass's delta1", "Glass's delta2",
    "r", or "all".

    Can be conducted in one step, or two steps:

        One step
        --------
        difference_test(formula_like, data).conduct(return_type = "Dataframe", effect_size = None)

        Two step
        --------
        model = difference_test(formula_like, data)
        model.conduct(return_type = "Dataframe", effect_size = None)

    """




    def __init__(self, formula_like, data = {}, conf_level = 0.95,
                 equal_variances = True, independent_samples = True,
                 wilcox_parameters = {"zero_method" : "wilcox", "correction" : False, "mode" : "auto"}, **keywords):



        # Determining which test to conduct
        if equal_variances == True and independent_samples == True:
            name = "Independent samples t-test"
        if equal_variances == True and independent_samples == False:
            name = "Paired samples t-test"
        if equal_variances == False and independent_samples == True:
            name = "Welch's t-test"
        if equal_variances == False and independent_samples == False:
            name = "Wilcoxon signed-rank test"
            parameters = {"zero_method" : "wilcox", "correction" : False, "mode" : "auto"}
            parameters.update(wilcox_parameters)



        self.DV, self.IV = patsy.dmatrices(formula_like + "- 1", data, 1)



        # Checking number of groups in IV, if > 2 stop function
        if len(self.IV.design_info.column_names) > 2:
            return print(" ",
                         "ERROR",
                         "The independent variables has more than 2 groups.",
                         "This method is not appropriate for this many groups.",
                         " ",
                         sep = "\n"*2)

        # Cleaning up category names from Patsy output
        categories = [re.findall(r"\[(.*)\]", name)[0] for name in self.IV.design_info.column_names]



        if name == "Wilcoxon signed-rank test":
            self.parameters = {"Test name" : name,
                               "Formula" : formula_like,
                               "Conf. Level": conf_level,
                               "Categories" : categories,
                               "Equal variances" : equal_variances,
                               "Independent samples" : independent_samples,
                               "Wilcox parameters" : parameters}
        else:
            self.parameters = {"Test name" : name,
                               "Formula" : formula_like,
                               "Conf. Level": conf_level,
                               "Categories" : categories,
                               "Equal variances" : equal_variances,
                               "Independent samples" : independent_samples}




    def conduct(self, return_type = "Dataframe", effect_size = None):

        # Parameter check
        if return_type.upper() not in ["DATAFRAME", "DICTIONARY"]:
            return print(" ",
                         "Not a supported return type. Only 'Dataframe' and 'Dictionary' are supported at this time.",
                         " ",
                         sep = "\n"*2)


        if effect_size != None:
            if type(effect_size) == str and effect_size != "all":
                effect_size = list(effect_size)
            elif type(effect_size) == str and effect_size == "all":
                effect_size = ["Cohen's D", "Hedge's G", "Glass's delta1", "Glass's delta2", "r"]

            for es in effect_size:
                if es  not in [None, "Cohen's D", "Hedge's G", "Glass's delta1", "Glass's delta2", "r", "all"]:
                    return print(" ",
                                 "Not a supported effect size. Either enter None or one of the following: 'Cohen's D', 'Hedge's G', 'Glass's delta1', 'Glass's delta2', 'r', and 'all'.",
                                 " ",
                                 sep = "\n"*2)



        # Splitting into seperate arrays and getting descriptive statistics
        group1, group2 = numpy.hsplit((self.DV * self.IV), 2)

        group1 = numpy.trim_zeros(group1)
        group2 = numpy.trim_zeros(group2)





        # Getting the summary table ready - part 1
        group1_info = summarize(group1, stats = ["N", "Mean", "SD", "SE", "Variance", "CI"], name = self.parameters["Categories"][0], ci_level = self.parameters["Conf. Level"], return_type = "Dictionary")
        group2_info = summarize(group2, stats = ["N", "Mean", "SD", "SE", "Variance", "CI"], name = self.parameters["Categories"][1], ci_level = self.parameters["Conf. Level"], return_type = "Dictionary")


        combined = summarize(numpy.vstack((group1, group2)),
                             stats = ["N", "Mean", "SD", "SE", "Variance", "CI"], name = "combined", ci_level = self.parameters["Conf. Level"], return_type = "Dictionary")


        diff = {}
        diff["Name"] = "diff"
        diff["Mean"]  = group1_info["Mean"] - group2_info["Mean"]





        # Performing the statistical test
        if self.parameters["Test name"] == "Independent samples t-test":
            stat, pval = scipy.stats.ttest_ind(group1, group2, nan_policy = 'omit')
            stat_name = "t"

            dof = group1_info["N"] + group2_info["N"] - 2

        if self.parameters["Test name"] == "Paired samples t-test":
            stat, pval = scipy.stats.ttest_rel(group1, group2, nan_policy = 'omit')
            stat_name = "t"

            dof = group1_info["N"] - 1

        if self.parameters["Test name"] == "Welch's t-test":
            stat, pval = scipy.stats.ttest_ind(group1, group2, equal_var = False, nan_policy = 'omit')
            stat_name = "t"

            ## Welch-Satterthwaite Degrees of Freedom ##
            dof = -2 + (((group1_info["Variance"]/group1_info["N"]) + (group2_info["Variance"]/group2_info["N"]))**2 / ((group1_info["Variance"]/group1_info["N"])**2 / (group1_info["N"]+1) + (group2_info["Variance"]/group2_info["N"])**2 / (group2_info["N"]+1)))


        if self.parameters["Test name"] == "Wilcoxon signed-rank test":
            d = group1 - group2
            d = numpy.reshape(d, (d.shape[0], ))

            stat, pval = scipy.stats.wilcoxon(d, zero_method = self.parameters["Wilcox parameters"]['zero_method'], correction = self.parameters["Wilcox parameters"]['correction'], mode = self.parameters["Wilcox parameters"]['mode'])
            stat_name = "W"

            dof = group1_info["N"] - 1


        # P value tails
        pval_lt = scipy.stats.t.cdf(stat, dof)
        pval_rt = 1 - scipy.stats.t.cdf(stat, dof)



        # Creating testing information table
        if self.parameters["Test name"] == "Wilcoxon signed-rank test":
            result_table = {self.parameters["Test name"] : [f"({self.parameters['Categories'][0]} = {self.parameters['Categories'][1]})",
                                                            f"{stat_name} =",
                                                            "Two sided p-value ="],
                            "Results" : ['',
                                         float(stat),
                                         float(pval)]}


        else:
            result_table = {self.parameters["Test name"] : [f"Difference ({self.parameters['Categories'][0]} - {self.parameters['Categories'][1]})",
                                                            "Degrees of freedom =",
                                                            f"{stat_name} =",
                                                            "Two sided test p-value =",
                                                            "Difference < 0 p-value =",
                                                            "Difference > 0 p-value ="
                                                            ],
                            "Results" : [float(diff["Mean"]),
                                         float(dof),
                                         float(stat),
                                         float(pval),
                                         float(pval_lt),
                                         float(pval_rt),

                                         ]}





        # Creating effect size table
        if effect_size != None:

            if self.parameters["Test name"] == "Wilcoxon signed-rank test":

                if effect_size != "r":
                    print(" ",
                          f"Only Point-Biserial r will be calulcated for the {self.parameters['Test name']}.",
                          " ",
                          sep = "\n"*2)

                r = stat / scipy.stats.rankdata(d).sum()

                result_table[self.parameters["Test name"]].append("Point-Biserial r")
                result_table["Results"].append(float(r))


            else:
                for es in effect_size:

                    if es == "Cohen's D":

                        if self.parameters["Test name"] == "Paired samples t-test":
                            #### DECIDE IF YOU WANT TO SUPPORT THIS VERSION - USED IN RESEARCHPY TTEST()
                            # Cohen's Dz (within-subjects design)
                            #d = stat / numpy.sqrt(group1_info["N"])
                            #result_table[self.parameters["Test name"]].append("Cohen's Dz")
                            #result_table["Results"].append(float(d))

                            # Cohen's Dav (within-subjects design)
                            d = (group1_info["Mean"] - group2_info["Mean"]) / ((group1_info["SD"] + group2_info["SD"]) / 2)
                            result_table[self.parameters["Test name"]].append("Cohen's Dav")
                            result_table["Results"].append(float(d))

                        else:
                            # Cohen's d (between-subjects desgn)
                            d = (group1_info['Mean'] - group2_info["Mean"]) / numpy.sqrt((((group1_info["N"] - 1)*group1_info["Variance"] + (group2_info["N"] - 1)*group2_info["Variance"]) / (group1_info["N"] + group2_info["N"] - 2)))


                            result_table[self.parameters["Test name"]].append("Cohen's Ds")
                            result_table["Results"].append(float(d))


                    if es == "Hedge's G":
                        if self.parameters["Test name"] == "Paired samples t-test":
                            # Cohen's Dz (within-subjects design)
                            #d = stat / numpy.sqrt(group1_info["N"])

                            # Cohen's Dav (within-subjects design)
                            d = (group1_info["Mean"] - group2_info["Mean"]) / ((group1_info["SD"] + group2_info["SD"]) / 2)
                            g = d * (1 - (3 / ((4*(group1_info["N"] + group2_info["N"])) - 9)))

                            result_table[self.parameters["Test name"]].append("Hedge's Gav")
                            result_table["Results"].append(float(g))

                        else:
                            # Cohen's d (between-subjects desgn)
                            d = (group1_info['Mean'] - group2_info["Mean"]) / numpy.sqrt((((group1_info["N"] - 1)*group1_info["Variance"] + (group2_info["N"] - 1)*group2_info["Variance"]) / (group1_info["N"] + group2_info["N"] - 2)))
                            g = d * (1 - (3 / ((4*(group1_info["N"] + group2_info["N"])) - 9)))

                            result_table[self.parameters["Test name"]].append("Hedge's G")
                            result_table["Results"].append(float(g))



                    if es == "Glass's delta1":
                        d1 = (group1_info["Mean"] - group2_info["Mean"]) / group1_info["SD"]

                        result_table[self.parameters["Test name"]].append("Glass's delta1")
                        result_table["Results"].append(float(d1))


                    if es == "Glass's delta2":
                        d2 = (group1_info["Mean"] - group2_info["Mean"]) / group2_info["SD"]

                        result_table[self.parameters["Test name"]].append("Glass's delta2")
                        result_table["Results"].append(float(d2))



                    if es == "r":
                        r = stat / numpy.sqrt(stat**2 + dof)

                        result_table[self.parameters["Test name"]].append("Point-Biserial r")
                        result_table["Results"].append(float(r))





        # Getting the summary table ready - part 2

        # Calculating standard error of the difference
        if self.parameters["Test name"] == "Welch's t-test":
            se_unpooled = numpy.sqrt( group1_info["Variance"] / group1_info["N"] +  group2_info["Variance"] / group2_info["N"]  )

            ci_lower_diff, ci_upper_diff = scipy.stats.t.interval(self.parameters["Conf. Level"],
                                                                  dof,
                                                                  loc = diff["Mean"],
                                                                  scale = se_unpooled)
            diff["SE"] = float(se_unpooled)


        elif self.parameters["Test name"] == "Paired samples t-test":
            difference = group1 - group2

            se = float(scipy.stats.sem(difference, nan_policy= 'omit'))

            ci_lower_diff, ci_upper_diff = scipy.stats.t.interval(self.parameters["Conf. Level"],
                                                                  dof,
                                                                  loc = diff["Mean"],
                                                                  scale = se)

            diff["SE"] = se
            diff["SD"] = float(difference.std(ddof = 1))



        else:
            var_pooled = ( (group1_info["N"] - 1)*group1_info["Variance"] + (group2_info["N"] - 1)*group2_info["Variance"] ) / (group1_info["N"] + group2_info["N"] - 2)

            se_pooled = numpy.sqrt( (var_pooled / group1_info["N"])  +  (var_pooled / group2_info["N"])    )

            ci_lower_diff, ci_upper_diff = scipy.stats.t.interval(self.parameters["Conf. Level"],
                                                                  dof,
                                                                  loc = diff["Mean"],
                                                                  scale = se_pooled)
            diff["SE"] = float(se_pooled)


        diff[f"{int(self.parameters['Conf. Level'] * 100)}% Conf."] = float(ci_lower_diff)
        diff["Interval"] = float(ci_upper_diff)



        group1_table = pandas.DataFrame.from_dict(group1_info, orient = 'index').T
        group2_table = pandas.DataFrame.from_dict(group2_info, orient = 'index').T
        combined_table = pandas.DataFrame.from_dict(combined, orient = 'index').T
        diff_table = pandas.DataFrame.from_dict(diff, orient = 'index').T

        summary_table = pandas.concat([group1_table, group2_table, combined_table, diff_table], ignore_index = True)
        summary_table.replace(numpy.nan, ' ', inplace = True)





        # Returning the information
        if return_type == "Dataframe":
            if self.parameters["Test name"] == "Wilcoxon signed-rank test":
                return summary_table.iloc[:-2, :], pandas.DataFrame(result_table)
            elif self.parameters["Test name"] == "Paired samples t-test":
                return summary_table.drop(2), pandas.DataFrame(result_table)
            else:
                return summary_table, pandas.DataFrame(result_table)

        if return_type == "Dictionary":
            if self.parameters["Test name"] == "Wilcoxon signed-rank test":
                return summary_table.iloc[:-2, :].to_dict(), result_table
            elif self.parameters["Test name"] == "Paired samples t-test":
                return summary_table.drop(2).to_dict(), result_table
            else:
                return summary_table.to_dict(), result_table
