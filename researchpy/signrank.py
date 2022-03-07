import pandas
import numpy
import scipy.stats



class signrank(object):


    def __init__(self, group1, group2, zero_method = "pratt", correction = False, mode = "auto"):

        self.group1 = group1
        self.group2 = group2

        self.zero_method = zero_method
        self.correction = correction
        self.mode = mode

        # Parameter Checks #
        if zero_method not in ["pratt", "wilcox"]:
            return print(" ",
                         "Only 'pratt' and 'wilcox' methods are supported."
                         " ",
                         sep = "\n"*2)

        if mode not in ["auto", "exact", "approx"]:
            return print(" ",
                         "Only 'auto', 'exact', and 'approx' calculations are supported."
                         " ",
                         sep = "\n"*2)



    def conduct(self, return_type = "Dataframe", effect_size = []):
        """


        Parameters
        ----------
        return_type : String, optional
            The data structure in which the results should be returned, either "Dictionary" or "Dataframe". The default is "Dataframe".

        effect_size : String or List for multiple effect sizes.

        Returns
        -------
        descriptives : Dictionary or Pandas DataFrame depending on return_type.
            A data structure containing the descriptive information regarding the ranked-signs.

        variance : Dictionary or Pandas DataFrame depending on return_type.
            A data structure containing the variance information regarding the ranked-sign test.

        results : Dictionary or Pandas DataFrame depending on return_type.
            A data structure containing the z-statistic, w-statistic, and p-value of the ranked-sign test.

        """

        ## Parameter Check ##
        if return_type.upper() not in ["DATAFRAME", "DICTIONARY"]:

            return print(" ",
                         "Not a supported return type. Only 'Dataframe' and 'Dictionary' are supported at this time.",
                         " ",
                         sep = "\n"*2)



        if type(effect_size) != list:
            return print(" ",
                         "The effect_size parameter must be in provided as a list.",
                         " ",
                         sep = "\n"*2)

        for es in effect_size:
            if es not in ["pb", "pearson"]:
                return print(" ",
                         "Only 'pb', and 'pearson' are supported at this time.",
                         " ",
                         sep = "\n"*2)




        # Calculating the differences between groups
        difference = self.group1 - self.group2


        if self.zero_method == "pratt":


            difference_abs = numpy.abs(difference)

            # Calculating Signed Information
            total_n = difference.shape[0]
            positive_n = difference[difference > 0].shape[0]
            negative_n = difference[difference < 0].shape[0]
            zero_n = difference[difference == 0].shape[0]

        else:

            difference = difference[difference != 0]
            difference_abs = numpy.abs(difference)

            # Calculating Signed Information
            total_n = difference.shape[0]
            positive_n = difference[difference > 0].shape[0]
            negative_n = difference[difference < 0].shape[0]
            zero_n = difference[difference == 0].shape[0]





        # Ranking the absolute difference |d|
        ranked = scipy.stats.rankdata(difference_abs)

        sign = numpy.where(difference < 0, -1, 1)

        ranked_sign = (sign * ranked)

        # Descriptive Information #
        total_sum_ranks = ranked.sum()
        positive_sum_ranks = ranked[difference > 0].sum()
        negative_sum_ranks = ranked[difference < 0].sum()
        zero_sum_ranks = ranked[difference == 0].sum()


        ## Dropping the Rank of the Zeros
        sign2 = numpy.where(difference == 0, 0, sign)
        ranked2 = sign2 * ranked
        ranked2 = numpy.where(difference == 0, 0, ranked2)


        # Expected T
        T = (sign * ranked_sign).sum()

        # Observered T
        T_obs = (sign2 * ranked2).sum()


        # Expected T+ and T-
        exp_positive = T_obs / 2
        exp_negative = T_obs / 2
        exp_zero = T - T_obs

        var_adj_T = (ranked2 * ranked2).sum()


        e_T_pos = total_n  * (total_n  + 1) / 4
        var_adj_T_pos = (1/4) * var_adj_T

        var_unadj_T_pos = ((total_n * (total_n + 1)) * (2 * total_n + 1)) /24
        var_zero_adj_T_pos = -1 * ((zero_n * (zero_n + 1)) * (2 * zero_n + 1)) /24

        var_ties_adj = var_adj_T_pos - var_unadj_T_pos - var_zero_adj_T_pos


        z = (positive_sum_ranks - exp_positive) / numpy.sqrt(var_adj_T_pos)

        w, pval = scipy.stats.wilcoxon(self.group1, self.group2,
                                       zero_method = self.zero_method,
                                       correction = self.correction,
                                       mode = self.mode)


        ##### Putting everything together #####

        ### Descriptive table
        descriptives = {"sign" : ["positive", "negative", "zero", "all"],
                        "obs" : [positive_n, negative_n, zero_n, total_n],
                        "sum ranks" : [positive_sum_ranks, negative_sum_ranks, zero_sum_ranks, total_sum_ranks],
                        "expected" : [exp_positive, exp_negative, exp_zero, T]}

        ##### Variance table
        variance = {"unadjusted variance" : var_unadj_T_pos,
                    "adjustment for ties" : var_ties_adj,
                    "adjustment for zeros" : var_zero_adj_T_pos,
                    "adjusted variance" : var_adj_T_pos}

        ##### Results table
        results = {"z" : z,
                   "w" : w,
                   "pval" : pval}


        if len(effect_size) != 0:

            for es in effect_size:

                if es == "pb":
                    results["Rank-Biserial r"] = results["w"] / descriptives["sum ranks"][-1]
                if es == "pearson":
                    results["Pearson r"] = results["z"] / numpy.sqrt(descriptives["obs"][-1])







        self.descriptives = descriptives
        self.variance = variance
        self.results = results


        if return_type == "Dataframe":
            descriptives = pandas.DataFrame.from_dict(descriptives)
            variance = pandas.DataFrame.from_dict(variance, orient = 'index').T
            results = pandas.DataFrame.from_dict(results, orient = 'index').T



        return (descriptives, variance, results)