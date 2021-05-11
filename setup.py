# -*- coding: utf-8 -*-
"""
@author: Corey Bryant


"""

from setuptools import setup

setup(name= "researchpy",
      version= "0.3.2",
      description= "Researchpy produces Pandas DataFrames that contain relevant statistical testing information that is commonly required for academic research.",
      long_description= "Researchpy produces Pandas DataFrames that contain relevant statistical testing information that is commonly required for academic research. The information is returned as Pandas DataFrames to make for quick and easy exporting of results to any format/method that works with the traditional Pandas DataFrame. researchpy is essentially a wrapper that combines various established packages such as pandas, scipy.stats, and statsmodels to get all the standard required information in one method. If analyses were not available in these packages, code was developed to fill the gap.",
      url= "http://researchpy.readthedocs.io/",
      author= "Corey Bryant",
      author_email= "CoreyBryant10@gmail.com",
      license = "MIT",
      packages= ['researchpy'],
      install_requires= ['scipy', 'numpy', 'pandas', 'statsmodels', 'patsy']
      )
