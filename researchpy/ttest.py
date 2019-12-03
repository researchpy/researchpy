# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 13:18:39 2018
Updated on December 23, 2018

@author: Corey Bryant
   
"""

import pandas
import numpy
import scipy.stats


def ttest(group1, group2, group1_name= None, group2_name= None, equal_variances= True, paired= False, correction= None):
    
    # Joining groups for table and calculating group mean difference 
    groups = group1.append(group2, ignore_index= True)
    groups_diff = numpy.mean(group1) - numpy.mean(group2)
    
    
    
    # Deciding which test to use
    if equal_variances == True and paired == False:
        test = "Independent t-test"
        t_val, p_val = scipy.stats.ttest_ind(group1, group2, nan_policy= 'omit')
        dof = group1.count() + group2.count() - 2
        
        # Confidence interval
        lt_p_val = scipy.stats.t.cdf(t_val, dof)
        rt_p_val = 1 - scipy.stats.t.cdf(t_val, dof)
        
        # Effect sizes
        # Cohen's d
        # Calculated using t*square root(1/group1_n + 1/group2_n)
        d = t_val* numpy.sqrt(1/group1.count() + 1/group2.count())
        
        # Common language effect size
        # Calculated z = |group1_mean - group2_mean| / square root (group1_variance + group2_variance)
        cles = 0 - abs(group1.mean() - group2.mean()) / numpy.sqrt(group1.var() + group2.var())
        
        # Hedge's G
        #Calculated as g = d * (1 - 3 / 4(group1_n + group2_n) - 9)
        g = d * (1 - (3 / ((4*(group1.count() + group2.count())) - 9)))
        
        # Glass's delta
        delta = (group1.mean() - group2.mean()) / group1.std()
        
        # r
        r = numpy.sqrt(abs(t_val)**2 / ((abs(t_val)**2) + dof))
        
        
        
    elif equal_variances == False and paired == False:   
        test = "Welch's t-test"
        t_val, p_val = scipy.stats.ttest_ind(group1, group2, equal_var = False, 
                                             nan_policy= 'omit')
        ## Welch-Satterthwaite Degrees of Freedom ##
        dof = ((group1.var()/group1.count()) + (group2.var()/group2.count()))**2 / ((group1.var()/group1.count())**2 / (group1.count()-1) + (group2.var()/group2.count())**2 / (group2.count()-1))  
   
        # Confidence interval
        lt_p_val = scipy.stats.t.cdf(t_val, dof)
        rt_p_val = 1 - scipy.stats.t.cdf(t_val, dof)
        
        # Effect size 
        # Cohen's d
        # USING THE T VALUE IN CALCULATION IS FAST THAN FULL FORMULA
        #d = (group1.mean() - group2.mean()) / numpy.sqrt(((group1.count() - 1) * group1.std()**2 + (group2.count() - 1) * group2.std()**2) / (group1.count() + group2.count() - 2))
        
        # Calculated using t*square root(1/group1_n + 1/group2_n)
        d = t_val* numpy.sqrt(1/group1.count() + 1/group2.count())
        
        # Hedge's G
        #Calculated as g = d * (1 - 3 / 4(group1_n + group2_n) - 9)
        g = d * (1 - (3 / ((4*(group1.count() + group2.count())) - 9)))
        
        # Glass's delta
        delta = (group1.mean() - group2.mean()) / group1.std()
        
        # Common language effect size
        # Calculated z = |group1_mean - group2_mean| / square root (group1_variance + group2_variance)
        cles = 0 - abs(group1.mean() - group2.mean()) / numpy.sqrt(group1.var() + group2.var())
        
        # r
        r = numpy.sqrt(abs(t_val)**2 / ((abs(t_val)**2) + dof))
        
        
        
    elif equal_variances == True and paired == True:
        group1 = group1[(group1.notnull()) & (group2.notnull())]
        group2 = group2[(group1.notnull()) & (group2.notnull())]
        
        groups = group1.append(group2, ignore_index= True)
        groups_diff = numpy.mean(group1) - numpy.mean(group2)
        diff = group1 - group2
        
        test = "Paired samples t-test"
        t_val, p_val = scipy.stats.ttest_rel(group1, group2)
        dof = group1.count() - 1
        
        # Confidence interval
        lt_p_val = scipy.stats.t.cdf(t_val, dof)
        rt_p_val = 1 - scipy.stats.t.cdf(t_val, dof)
                
        # Effect sizes 
        # Cohen's d
        # Using formula provided by Rosenthal (1991)
        # d = t / square root(n)
        d = t_val / numpy.sqrt(diff.count())
        
        # Hedge's G
        #Calculated as g = d * (1 - 3 / 4(group1_n + group2_n) - 9)
        g = d * (1 - (3 / ((4*(group1.count() + group2.count())) - 9)))
        
        # Glass's delta
        delta = (group1.mean() - group2.mean()) / group1.std()
        
        # r
        r = numpy.sqrt(abs(t_val)**2 / ((abs(t_val)**2) + dof))
        
        
        
    elif equal_variances == False and paired == True:
        test = "Wilcoxon signed-rank test"
            
        if correction == None:
            correction = False
        else:
            correction = correction
        
        group1 = group1[(group1.notnull()) & (group2.notnull())]
        group2 = group2[(group1.notnull()) & (group2.notnull())]
        
        groups_diff = group1 - group2
        groups_diff_non0 = groups_diff[groups_diff != 0]
 
        ## t_bar Calculated as t_bar = n(n+1) / 4
        ## where n is number of non-zero observations
        t_bar = groups_diff_non0.count() * (groups_diff_non0.count() + 1) / 4

        ## SE_t_bar calculated as
        ## se_t_bar = square_root(n(n + 1)(2n + 1)/24),
        ## where n is number of non-zero observations
        se_t_bar = numpy.sqrt(groups_diff_non0.count() * (groups_diff_non0.count() + 1) * ((2 * groups_diff_non0.count()) + 1) / 24 )


        t_val, p_val = scipy.stats.wilcoxon(group1, group2,
                                            correction= f"{correction}")
        
        dof = group1.count() - 1 

        ## Calculating z score; using formula from Rosenthal, 1991
        ## z = t_val - t_bar / se_t_bar
        z = (t_val - t_bar) / se_t_bar

        ## Effect size
        ##  r = z / square_root(N)
        r = z / numpy.sqrt(group1.count() + group2.count())
        
        
        
        

    #### PUTTING THE INFORMATION INTO A DATAFRAME #####
    table = pandas.DataFrame(numpy.zeros(shape= (3,7)), 
                         columns = ['Variable', 'N', 'Mean', 'SD', 'SE',
                                    '95% Conf.', 'Interval'])
    
    # Setting up the first column (Variable names)
    if group1_name != None:
        group1_name = group1_name
    else:
        group1_name = group1.name
    
    if group2_name != None:
        group2_name = group2_name
    else:
        group2_name = group2.name
    
    table.iloc[0,0] = group1_name
    table.iloc[1,0] = group2_name
    
    if test == "Paired samples t-test":
        table.iloc[2,0] = 'diff'
    else:
        table.iloc[2,0] = 'combined'
    
    # Setting up the second column (Number of observations)
    table.iloc[0,1] = group1.count()              
    table.iloc[1,1] = group2.count()
    if test == "Paired samples t-test":
        table.iloc[2,1] = diff.count()
    else:
        table.iloc[2,1] = groups.count()
        
    # Setting up the third column (Mean)
    table.iloc[0,2] = numpy.mean(group1)              
    table.iloc[1,2] = numpy.mean(group2)
    if test == "Paired samples t-test":
        table.iloc[2,2] = numpy.mean(diff)
    else:
        table.iloc[2,2] = numpy.mean(groups)
    
    # Setting up the fourth column (Standard Deviation (SD))
    table.iloc[0,3] = numpy.std(group1, ddof= 1)          
    table.iloc[1,3] = numpy.std(group2, ddof= 1)
    if test == "Paired samples t-test":
        table.iloc[2,3] = numpy.std(diff, ddof= 1)
    else:
        table.iloc[2,3] = numpy.std(groups, ddof= 1)
    
    # Setting up the fith column (Standard Error (SE))
    table.iloc[0,4] = scipy.stats.sem(group1, nan_policy= 'omit')          
    table.iloc[1,4] = scipy.stats.sem(group2, nan_policy= 'omit')
    if test == "Paired samples t-test":
        table.iloc[2,4] = scipy.stats.sem(diff, nan_policy= 'omit')
    else:
        table.iloc[2,4] = scipy.stats.sem(groups, nan_policy= 'omit')
    
    # Setting up the sixth and seventh column (95% CI)
    table.iloc[0,5], table.iloc[0,6] = scipy.stats.t.interval(0.95, 
                                          group1.count() - 1, 
                                          loc= numpy.mean(group1), 
                                          scale= scipy.stats.sem(group1, nan_policy= 'omit'))
    table.iloc[1,5], table.iloc[1,6] = scipy.stats.t.interval(0.95, 
                                          group2.count() - 1, 
                                          loc= numpy.mean(group2), 
                                          scale= scipy.stats.sem(group2, nan_policy= 'omit'))
    if test == "Paired samples t-test":
        table.iloc[2,5], table.iloc[2,6] = scipy.stats.t.interval(0.95, 
                                          diff.count() - 1, 
                                          loc= numpy.mean(diff), 
                                          scale= scipy.stats.sem(diff, nan_policy= 'omit'))
    else:
        table.iloc[2,5], table.iloc[2,6] = scipy.stats.t.interval(0.95, 
                                          groups.count() - 1, 
                                          loc= numpy.mean(groups), 
                                          scale= scipy.stats.sem(groups, nan_policy= 'omit'))
    
    if equal_variances == False and paired == True:
        table2 = pandas.DataFrame(numpy.zeros(shape= (6,2)), 
                         columns = ['Wilcoxon signed-rank test', 'results'])
        
        table2.iloc[0,0] = f"Mean for {group1_name} = "
        table2.iloc[0,1] = numpy.mean(group1)
        
        table2.iloc[1,0] = f"Mean for {group2_name} = "
        table2.iloc[1,1] = numpy.mean(group2)
        
        table2.iloc[2,0] = f"T value = "
        table2.iloc[2,1] = round(t_val, 4)
        
        table2.iloc[3,0] = f"Z value = "
        table2.iloc[3,1] = round(z, 4)
        
        table2.iloc[4,0] = f"Two sided p value = "
        table2.iloc[4,1] = round(p_val, 4)
        
        table2.iloc[5,0] = f"r = "
        table2.iloc[5,1] = round(r, 4)
        
        return table2
        
    else:
        table2 = pandas.DataFrame(numpy.zeros(shape= (10,2)), 
                         columns = ['Test', 'results'])
        
        table2.rename(columns= {'Test': f'{test}'}, inplace= True)
        
        table2.iloc[0,0] = f"Difference ({group1_name} - {group2_name}) = "
        table2.iloc[0,1] = round(groups_diff, 4)
        
        table2.iloc[1,0] = "Degrees of freedom = "
        table2.iloc[1,1] = round(dof, 4)
        
        table2.iloc[2,0] = "t = "
        table2.iloc[2,1] = round(t_val, 4)
        
        table2.iloc[3,0] = "Two side test p value = "
        table2.iloc[3,1] = round(p_val, 4)
        
        table2.iloc[4,0] = f"Difference < 0 p value = "
        table2.iloc[4,1] = round(lt_p_val, 4)

        table2.iloc[5,0] = f"Difference > 0 p value = "
        table2.iloc[5,1] = round(rt_p_val, 4)
        
        table2.iloc[6,0] = f"Cohen's d = "
        table2.iloc[6,1] = round(d, 4)
        
        table2.iloc[7,0] = f"Hedge's g = "
        table2.iloc[7,1] = round(g, 4)
        
        table2.iloc[8,0] = f"Glass's delta = "
        table2.iloc[8,1] = round(delta, 4)
        
        table2.iloc[9,0] = f"r = "
        table2.iloc[9,1] = round(r, 4)
    
        return table, table2
