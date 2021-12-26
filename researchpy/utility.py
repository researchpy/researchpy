
def rounder(lst, decimals = 4):
    """
        Iterates through a list and returns the rounded number.
    """

    idx = 0
    for item in lst:
        lst[idx] = round(item, decimals)
        idx += 1


def name_cleaner(lst):
    """
        Iterates through a list and removes 'C(' and ')' from the string then
        returns the cleaned name to the lst.
    """

    idx = 0
    for name in lst:
        lst[idx] = name.replace('C(', '').replace(')', '')
        idx += 1
