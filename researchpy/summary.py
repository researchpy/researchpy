# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 12:28:20 2018

@author: bryantcm


To do:
    - Add categorical data summary support

"""

import pandas
import numpy
import scipy.stats

## summary_cont() provides descriptive statistics for continuous data
def summary_cont(group1, conf = 0.95, decimals = 4):



    conf_level = f"{round(conf * 100)}%"

    if type(group1) == pandas.core.series.Series:

        #### PUTTING THE INFORMATION INTO A DATAFRAME #####
        table = pandas.DataFrame(numpy.zeros(shape= (1,7)),
                         columns = ['Variable', 'N', 'Mean', 'SD', 'SE', f'{conf_level} Conf.', 'Interval'])


        # Setting up the first column (Variable names)
        table.iloc[0,0] = group1.name

        # Setting up the second column (Number of observations)
        table.iloc[0,1] = group1.count()

        # Setting up the third column (Mean)
        table.iloc[0,2] = numpy.mean(group1)

        # Setting up the fourth column (Standard Deviation (SD))
        table.iloc[0,3] = numpy.std(group1, ddof= 1)

        # Setting up the fith column (Standard Error (SE))
        table.iloc[0,4] = scipy.stats.sem(group1, nan_policy= 'omit')

        # Setting up the sixth and seventh column (95% CI)
        table.iloc[0,5], table.iloc[0,6] = scipy.stats.t.interval(conf,
                                          group1.count() - 1,
                                          loc= numpy.mean(group1), scale= scipy.stats.sem(group1, nan_policy= 'omit'))

    elif type(group1) == pandas.core.frame.DataFrame:

        table = pandas.DataFrame(numpy.zeros(shape= (1,7)),
                         columns = ['Variable', 'N', 'Mean', 'SD', 'SE',
                                    f'{conf_level} Conf.', 'Interval'])

        count = 0

        for ix, df_col in group1.iteritems():

            count = count + 1

            if count == 0:

                table = pandas.DataFrame(numpy.zeros(shape= (1,7)),
                         columns = ['Variable', 'N', 'Mean', 'SD', 'SE',
                                    f'{conf_level} Conf.', 'Interval'])

                # Setting up the first column (Variable names)
                table.iloc[0,0] = ix

                # Setting up the second column (Number of observations)
                table.iloc[0,1] = df_col.count()

                # Setting up the third column (Mean)
                table.iloc[0,2] = numpy.mean(df_col)

                # Setting up the fourth column (Standard Deviation (SD))
                table.iloc[0,3] = numpy.std(df_col, ddof= 1)

                # Setting up the fith column (Standard Error (SE))
                table.iloc[0,4] = scipy.stats.sem(df_col, nan_policy= 'omit')

                # Setting up the sixth and seventh column (95% CI)
                table.iloc[0,5], table.iloc[0,6] = scipy.stats.t.interval(conf,
                                          df_col.count() - 1,
                                          loc= numpy.mean(df_col),
                                          scale= scipy.stats.sem(df_col, nan_policy= 'omit'))
            else:

                table_a = pandas.DataFrame(numpy.zeros(shape= (1,7)),
                         columns = ['Variable', 'N', 'Mean', 'SD', 'SE',
                                    f'{conf_level} Conf.', 'Interval'])

                # Setting up the first column (Variable names)
                table_a.iloc[0,0] = ix

                # Setting up the second column (Number of observations)
                table_a.iloc[0,1] = df_col.count()

                # Setting up the third column (Mean)
                table_a.iloc[0,2] = numpy.mean(df_col)

                # Setting up the fourth column (Standard Deviation (SD))
                table_a.iloc[0,3] = numpy.std(df_col, ddof= 1)

                # Setting up the fith column (Standard Error (SE))
                table_a.iloc[0,4] = scipy.stats.sem(df_col, nan_policy= 'omit')

                # Setting up the sixth and seventh column (95% CI)
                table_a.iloc[0,5], table_a.iloc[0,6] = scipy.stats.t.interval(conf,
                                          df_col.count() - 1,
                                          loc= numpy.mean(df_col),
                                          scale= scipy.stats.sem(df_col, nan_policy= 'omit'))



            table = pandas.concat([table, table_a], ignore_index= "true")

        table.drop(0, inplace= True)
        table.reset_index(inplace= True, drop= True)



    elif type(group1) == pandas.core.groupby.SeriesGroupBy:
        ## Validated with R

        cnt = group1.count()
        cnt.rename("N", inplace= True)
        mean = group1.mean()
        mean.rename("Mean", inplace= True)
        std = group1.std(ddof= 1)
        std.rename("SD", inplace= True)
        se = group1.sem()
        se.rename("SE", inplace= True)

        # 95% CI
        l_ci, u_ci = scipy.stats.t.interval(conf,
                                          cnt - 1,
                                          loc= group1.mean(),
                                          scale= group1.sem())

        table = pandas.concat([cnt, mean, std, se,
                               pandas.Series(l_ci, index = cnt.index),
                               pandas.Series(u_ci, index = cnt.index)],
                              axis= 'columns')

        table.rename(columns = {'count': 'N', 'mean': 'Mean', 'std': 'SD',
                                'sem': 'SE', 0 : f'{conf_level} Conf.', 1 : "Interval" },
                                inplace= True)


    elif type(group1) == pandas.core.groupby.DataFrameGroupBy :

        # There has to be a better way to get the lower and upper CI limits
        # in the groupby table. Until then, this works :/
        def l_ci(x):
            l_ci, _ = scipy.stats.t.interval(conf,
                                                x.count() - 1,
                                                loc= x.mean(),
                                                scale= x.sem())


            return l_ci

        def u_ci(x):
            _, u_ci = scipy.stats.t.interval(conf,
                                                x.count() - 1,
                                                loc= x.mean(),
                                                scale= x.sem())


            return u_ci

        table = group1.agg(['count', numpy.mean, numpy.std,
                            pandas.DataFrame.sem, l_ci, u_ci])

        table.rename(columns = {'count': 'N', 'mean': 'Mean', 'std': 'SD',
                                'sem': 'SE', "l_ci" : f'{conf_level} Conf.', "u_ci" : "Interval"}, inplace= True)


    else:
        return "This method only works with a Pandas Series, Dataframe, or Groupby object"


    print("\n")
    return table.round(decimals)








## summary_cat provides descriptives information for categorical data. It can
##  also handle numeric data since it's just counts and percents

def summary_cat(group1, ascending= False):

    if type(group1) == pandas.core.series.Series:
        table = group1.value_counts()
        table.rename("Count", inplace= True)
        table = pandas.DataFrame(table)

        table['Percent'] = ((table['Count']/table['Count'].sum()) * 100).round(2)

        if ascending == False:
            table.sort_values(by= 'Count', ascending= False, inplace= True)
        else:
            table.sort_values(by= 'Count', ascending= True, inplace= True)

        table['Variable'] = ""
        table.iloc[0,2] = group1.name

        table.reset_index(inplace= True)
        table.rename(columns= {'index': 'Outcome'}, inplace= True)
        table = table[['Variable', 'Outcome', 'Count', 'Percent']]

    elif type(group1) == pandas.core.frame.DataFrame:

        count = 0

        for ix, df_col in group1.iteritems():

            count = count + 1

            if count == 1:

                table = df_col.value_counts()
                table.rename("Count", inplace= True)
                table = pandas.DataFrame(table)

                table['Percent'] = ((table['Count']/table['Count'].sum()) * 100).round(2)

                if ascending == False:
                    table.sort_values(by= 'Count', ascending= False, inplace= True)
                else:
                    table.sort_values(by= 'Count', ascending= True, inplace= True)

                table['Variable'] = ""
                table.iloc[0,2] = ix

                table.reset_index(inplace= True)
                table.rename(columns= {'index': 'Outcome'}, inplace= True)
                table = table[['Variable', 'Outcome', 'Count', 'Percent']]

            else:

                table_c = df_col.value_counts()
                table_c.rename("Count", inplace= True)
                table_c = pandas.DataFrame(table_c)

                table_c['Percent'] = ((table_c['Count']/table_c['Count'].sum()) * 100).round(2)

                if ascending == False:
                    table_c.sort_values(by= 'Count', ascending= False, inplace= True)
                else:
                    table_c.sort_values(by= 'Count', ascending= True, inplace= True)

                table_c['Variable'] = ""
                table_c.iloc[0,2] = ix

                table_c.reset_index(inplace= True)
                table_c.rename(columns= {'index': 'Outcome'}, inplace= True)
                table_c = table_c[['Variable', 'Outcome', 'Count', 'Percent']]

                table = pandas.concat([table, table_c], ignore_index= "true")

    else:
        print("This method can only be used with Pandas Series or DataFrames")

    return table



def codebook(data):
    """
    This function returns descriptive information about the variables at hand.
    Accepts Pandas Series or Pandas DataFrame objects.
    """



    if type(data) == pandas.core.series.Series:
        """

        How to provide summary information for Series objects

        """



        if "int" in str(data.dtype) or "float" in str(data.dtype):
            print(f"Variable: {data.name}    Data Type: {data.dtype}", "\n")

            print(f" Number of Obs.: {data.size} \n",
                  f"Number of missing obs.: {data.size - data.count()} \n",
                  f"Percent missing: {((data.size - data.count()) / data.size * 100).round(2)} \n",
                  f"Number of unique values: {data.nunique()} \n")

            print(f" Range: [{data.min()}, {data.max()}] \n",
                  f"Mean: {round(data.mean(), 2)} \n",
                  f"Standard Deviation: {round(data.std(), 2)} \n",
                  f"Mode: {data.mode()[0]} \n",
                  f"10th Percentile: {data.quantile(.10, interpolation= 'linear')} \n",
                  f"25th Percentile: {data.quantile(.25, interpolation= 'linear')} \n",
                  f"50th Percentile: {data.quantile(.50, interpolation= 'linear')} \n",
                  f"75th Percentile: {data.quantile(.75, interpolation= 'linear')} \n",
                  f"90th Percentile: {data.quantile(.90, interpolation= 'linear')} \n",)

            print("\n" * 3)



        elif "object" in str(data.dtype) or "category" == data.dtype:
            tab = dict(data.value_counts())
            tab = dict(sorted(tab.items()))
            tab = {"Values" : list(tab.keys()), "Frequency" : list(tab.values())}
            tab = pandas.DataFrame(tab)


            print(f"Variable: {data.name}    Data Type: {data.dtype}", "\n")

            print(f" Number of Obs.: {data.size} \n",
                  f"Number of missing obs.: {data.size - data.count()} \n",
                  f"Percent missing: {((data.size - data.count()) / data.size * 100).round(2)} \n",
                  f"Number of unique values: {data.nunique()} \n")

            print(f" Data Values and Counts: \n \n",
                  tab.to_string(index = False))

            print("\n" * 3)



        elif "datetime" in str(data.dtype):
            print(f"Variable: {data.name}    Data Type: {data.dtype}", "\n")

            print(f" Number of Obs.: {data.size} \n",
                  f"Number of missing obs.: {data.size - data.count()} \n",
                  f"Percent missing: {((data.size - data.count()) / data.size * 100).round(2)} \n",
                  f"Number of unique values: {data.nunique()} \n")

            print(f" Range: [{data.min()}, {data.max()}]")

            print("\n" * 3)




        else:
            print(f"type(data) is not supported at this time.")

            print("\n" * 3)





    elif type(data) == pandas.core.frame.DataFrame:
        """

        How to provide summary information for DataFrame objects

        """

        for col in data.columns:

            if "int" in str(data[col].dtype) or "float" in str(data[col].dtype):
                print(f"Variable: {data[col].name}    Data Type: {data[col].dtype}", "\n")

                print(f" Number of Obs.: {data[col].size} \n",
                      f"Number of missing obs.: {data[col].size - data[col].count()} \n",
                      f"Percent missing: {((data[col].size - data[col].count()) / data[col].size * 100).round(2)} \n",
                      f"Number of unique values: {data[col].nunique()} \n")

                print(f" Range: [{data[col].min()}, {data[col].max()}] \n",
                      f"Mean: {round(data[col].mean(), 2)} \n",
                      f"Standard Deviation: {round(data[col].std(), 2)} \n",
                      f"Mode: {data[col].mode()[0]} \n",
                      f"10th Percentile: {data[col].quantile(.10, interpolation= 'linear')} \n",
                      f"25th Percentile: {data[col].quantile(.25, interpolation= 'linear')} \n",
                      f"50th Percentile: {data[col].quantile(.50, interpolation= 'linear')} \n",
                      f"75th Percentile: {data[col].quantile(.75, interpolation= 'linear')} \n",
                      f"90th Percentile: {data[col].quantile(.90, interpolation= 'linear')} \n")

                print("\n" * 3)



            elif "object" in str(data[col].dtype) or "category" == data[col].dtype:
                tab = dict(data[col].value_counts())
                tab = dict(sorted(tab.items()))
                tab = {"Values" : list(tab.keys()), "Frequency" : list(tab.values())}
                tab = pandas.DataFrame(tab)


                print(f"Variable: {data[col].name}    Data Type: {data[col].dtype}", "\n")

                print(f" Number of Obs.: {data[col].size} \n",
                      f"Number of missing obs.: {data[col].size - data[col].count()} \n",
                      f"Percent missing: {((data[col].size - data[col].count()) / data[col].size * 100).round(2)} \n",
                      f"Number of unique values: {data[col].nunique()} \n")

                print(f" Data Values and Counts: \n \n",
                      tab.to_string(index = False))

                print("\n" * 3)



            elif "datetime" in str(data[col].dtype):
                print(f"Variable: {data[col].name}    Data Type: {data[col].dtype}", "\n")

                print(f" Number of Obs.: {data[col].size} \n",
                      f"Number of missing obs.: {data[col].size - data[col].count()} \n",
                      f"Percent missing: {((data[col].size - data[col].count()) / data[col].size * 100).round(2)} \n",
                      f"Number of unique values: {data[col].nunique()} \n")

                print(f" Range: [{data[col].min()}, {data[col].max()}]")

                print("\n" * 3)




            else:
                print(f"{data.dtype} is not supported at this time.")

                print("\n" * 3)





    else:
        print(f"Current data type, {type(data)}, is not supported. Currently, only Pandas Series and DataFrame are supported.")






def summarize(data = {}, name = None, stats = [], ci_level = 0.95, return_type = "Dataframe"):

    """

    Available options are ["N", "Mean", "Median", "Variance", "SD", "SE", "CI",
                           'Min', 'Max', 'Range', "Kurtosis", "Skew"].

    The default returned is ["N", "Mean", "Median", "Variance", "SD", "SE", "CI"] at a 95% Conf. Interval.


    """
    # Parameter check
    if return_type.upper() not in ["DATAFRAME", "DICTIONARY"]:
        return print(" ",
                     "Not a supported return type. Only 'Dataframe' and 'Dictionary' are supported at this time.",
                     " ",
                     sep = "\n"*2)



    if len(stats) == 0:
        stats = ["N", "Mean", "Median", "Variance", "SD", "SE", "CI"]

    results = {}

    if name != None:
        results["Name"] = name

    if name == None:
        try:
            results["Name"] = data.name
        except:
            ...

    if "N" in stats:
        results["N"] = float(numpy.count_nonzero(~numpy.isnan(data)))

    if "Mean" in stats:
        results["Mean"] = float(numpy.nanmean(data))

    if "Median" in stats:
        results["Median"] = float(numpy.nanmedian(data))

    if "Variance" in stats:
        results["Variance"] = float(numpy.nanvar(data, ddof = 1))

    if "SD" in stats:
        results["SD"] = float(numpy.nanstd(data, ddof = 1))

    if "SE" in stats:
        results["SE"] = float(scipy.stats.sem(data, nan_policy= 'omit'))

    if "CI" in stats:
        ci_lower, ci_upper = scipy.stats.t.interval(ci_level,
                                                    int(numpy.isfinite(data).sum()) - 1,
                                                    loc = numpy.nanmean(data),
                                                    scale = scipy.stats.sem(data, nan_policy= 'omit'))

        results[f"{int(ci_level * 100)}% Conf."] = float(ci_lower)
        results["Interval"] = float(ci_upper)

    if "Min" in stats:
        results["Min"] = float(numpy.nanmin(data))

    if "Max" in stats:
        results["Max"] = float(numpy.nanmax(data))

    if "Range" in stats:
        results["Range"] = float(numpy.nanmax(data) - numpy.nanmin(data))

    if "Kurtosis" in stats:
        # This computes kurtosis using Pearson's definition
        results["Kurtosis"] = float(scipy.stats.kurtosis(data, fisher = False, nan_policy = 'omit'))

    if "Skew" in stats:
        results["Skew"] = float(scipy.stats.skew(data, nan_policy = 'omit'))



    if return_type == "Dataframe":
        return pandas.DataFrame.from_dict(results, orient='index').T
    elif return_type == "Dictionary":
        return results
