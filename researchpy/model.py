


class model():
    """

    This is the base -model- object for Researchpy. By default, missing
    observations are dropped from the data. -matrix_type- parameter determines
    which design matrix will be returned; value of 1 will return a design matrix
    with the intercept, while a value of 0 will not.

    """

    def __init__(self, formula_like, data = {}, matrix_type = 1):
        # matrix_type = 1 includes intercept
        # matrix_type = 0 does not include the intercept



        if matrix_type == 1:
            self.DV, self.IV = patsy.dmatrices(formula_like, data, 1)
        if matrix_type == 0:
            self.DV, self.IV = patsy.dmatrices(formula_like + "- 1", data, 1)

        self.nobs = self.IV.shape[0]

        # Model design information
        self.formula = formula_like
        self._DV_design_info  = self.DV.design_info
        self._IV_design_info  = self.IV.design_info



        
