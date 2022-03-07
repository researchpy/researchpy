
def rounder(lst, decimals = 4):
    """
        Iterates through a list and returns the rounded number.
    """

    idx = 0
    for item in lst:
        lst[idx] = round(item, decimals)
        idx += 1





def patsy_column_cleaner(factor):
    """

    Parameters
    ----------
    factor : A list of factors and levels which is provided by Patsy's design_info.column_names object.


    Returns
    -------
    String


    Description
    -----------
    This function returns a level name as a string.

    Ex.
    column_names = ['Intercept',
                    'C(drug, Treatment(2))[T.1]',
                    'C(drug, Treatment(2))[T.3]',
                    'C(drug, Treatment(2))[T.4]',
                    'disease',
                    'C(drug, Treatment(2))[T.1]:disease',
                    'C(drug, Treatment(2))[T.3]:disease',
                    'C(drug, Treatment(2))[T.4]:disease']


    [IN]  [patsy_column_cleaner(level) for level in column_names]
    [OUT] ['Intercept', '1', '3', '4', 'disease', '1:disease', '3:disease', '4:disease']


    """

    treatment_reference_pattern = re.compile(r'(?<=Treatment\()(.*?)(?=\))')
    factor_pattern = re.compile(r'(?<=C\()(.*?)(?=,|\))')
    level_pattern = re.compile(r'(?<=\[..)(.*?)(?=\])')


    factor = factor.split(":")


    # This renames non-interaction term names
    if len(factor) == 1:
        # This renames categorical variables
        if factor[0].startswith("C("):

            var_name = re.findall(level_pattern, factor[0])
            return var_name[0]

        else:
            # This keeps non-categorical variables names the same
            return factor[0]
    else:

        var_name = []

        for level in factor:
            if "C(" in level:

                var_name.append(''.join(re.findall(level_pattern, level)))
            else:
                var_name.append(level)

        return ':'.join(var_name)





def patsy_term_cleaner(factor):
    """


    Parameters
    ----------
    factor : A list of factors and levels which is provided by Patsy's design_info.term_names object.

    Returns
    -------
    String


    Description
    -----------
    This function returns a level name as a string.

    Ex.
    term_names = ['Intercept',
                  'C(drug, Treatment(2))',
                  'disease',
                  'C(drug, Treatment(2)):disease']


    [IN]  [patsy_term_cleaner(term) for term in term_names]
    [OUT] ['Intercept', 'drug', 'disease', 'drug:disease']

    """

    factor_pattern = re.compile(r'(?<=C\()(.*?)(?=,|\))')

    factor = factor.split(":")


    # This renames non-interaction term names
    if len(factor) == 1:
        # This renames categorical variables
        if factor[0].startswith("C("):

            var_name = ''.join(re.findall(factor_pattern, factor[0]))
            return var_name

        else:
            # This keeps non-categorical variables names the same
            return factor[0]
    else:

        var_name = []

        for level in factor:
            if "C(" in level:

                var_name.append(''.join(re.findall(factor_pattern, level)))
            else:
                var_name.append(level)

        return ':'.join(var_name)
