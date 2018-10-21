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
def summary_cont(group1):

    if type(group1) == pandas.core.series.Series:

        #### PUTTING THE INFORMATION INTO A DATAFRAME #####
        table = pandas.DataFrame(numpy.zeros(shape= (1,7)),
                         columns = ['Variable', 'N', 'Mean', 'SD', 'SE',
                                    '95% Conf.', 'Interval'])

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
        table.iloc[0,5], table.iloc[0,6] = scipy.stats.t.interval(0.95,
                                          group1.count() - 1,
                                          loc= numpy.mean(group1),
                                          scale= scipy.stats.sem(group1, nan_policy= 'omit'))

    elif type(group1) == pandas.core.frame.DataFrame:

        table = pandas.DataFrame(numpy.zeros(shape= (1,7)),
                         columns = ['Variable', 'N', 'Mean', 'SD', 'SE',
                                    '95% Conf.', 'Interval'])
       # table.drop(0, inplace= True)

        count = 0

        for ix, df_col in group1.iteritems():

            count = count + 1

            if count == 0:

                table = pandas.DataFrame(numpy.zeros(shape= (1,7)),
                         columns = ['Variable', 'N', 'Mean', 'SD', 'SE',
                                    '95% Conf.', 'Interval'])

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
                table.iloc[0,5], table.iloc[0,6] = scipy.stats.t.interval(0.95,
                                          df_col.count() - 1,
                                          loc= numpy.mean(df_col),
                                          scale= scipy.stats.sem(df_col, nan_policy= 'omit'))
            else:

                table_a = pandas.DataFrame(numpy.zeros(shape= (1,7)),
                         columns = ['Variable', 'N', 'Mean', 'SD', 'SE',
                                    '95% Conf.', 'Interval'])

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
                table_a.iloc[0,5], table_a.iloc[0,6] = scipy.stats.t.interval(0.95,
                                          df_col.count() - 1,
                                          loc= numpy.mean(df_col),
                                          scale= scipy.stats.sem(df_col, nan_policy= 'omit'))



            table = pandas.concat([table, table_a], ignore_index= "true")

        table.drop(0, inplace= True)
        table.reset_index(inplace= True, drop= True)



    elif type(group1) == pandas.core.groupby.SeriesGroupBy:

        cnt = group1.count()
        cnt.rename("N", inplace= True)
        mean = group1.mean()
        mean.rename("Mean", inplace= True)
        std = group1.std(ddof= 1)
        std.rename("SD", inplace= True)
        se = group1.sem()
        #se = group1.apply(lambda x: scipy.stats.sem(x, nan_policy= 'omit'))
        #se = scipy.stats.sem(group1, nan_policy= 'omit')
        se.rename("SE", inplace= True)

        # 95% CI
        l_ci = mean - (1.960 * (std/numpy.sqrt(cnt-1)))
        u_ci = mean + (1.960 * (std/numpy.sqrt(cnt-1)))


        table = pandas.concat([cnt, mean, std, se, l_ci, u_ci],
                              axis= 'columns')

        table.rename(columns = {'count': 'N', 'mean': 'Mean', 'std': 'SD', 
                                'sem': 'SE', 0 : "95% Conf.", 1 : "Interval" }, 
                                inplace= True)


    elif type(group1) == pandas.core.groupby.groupby.DataFrameGroupBy :

        l_ci = lambda x: numpy.mean(x) - (1.960 * (numpy.std(x)/numpy.sqrt(x.count() - 1)))
        l_ci.__name__ = "95% Conf."

        u_ci = lambda x: numpy.mean(x) + (1.960 * (numpy.std(x)/numpy.sqrt(x.count() - 1)))
        u_ci.__name__ = "Interval"

        table = group1.agg(['count', numpy.mean, numpy.std,
                            pandas.DataFrame.sem, l_ci, u_ci])
    
        table.rename(columns = {'count': 'N', 'mean': 'Mean', 'std': 'SD', 
                                'sem': 'SE'}, inplace= True)


    else:
        return "This method only works with a Pandas Series, Dataframe, or Groupby object"


    print("\n")
    return table








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
