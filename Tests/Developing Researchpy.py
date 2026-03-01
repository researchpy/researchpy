"""

In a command prompt (internal to Pychamr works also) activate the virtual environment for the
current version of researchpy and then install researchpy local version.

1. Activate virtual environment
    C:\Users\Servi\.virtualenvs\researchpy\v0.4.0\Scripts\activate

2. Install local version of researchpy in development mode
    python -m pip install -e .

"""
# %%
import researchpy as rp
import pandas as pd
pd.set_option('display.max_columns',30,
              'display.width',1000)
import numpy as np



url = "https://stats.idre.ucla.edu/stat/stata/dae/binary.dta"
pol = pd.read_stata(url)

m = rp.LogisticRegression("admit ~ gre + gpa + C(rank)", data=pol)
m.table_regression_results()


m2 = rp.ols("admit ~ gre + gpa + C(rank)", data=pol)
m2.table_regression_results()



# %%
url = "https://stats.idre.ucla.edu/stat/data/hsb2.dta"
hsb = pd.read_stata(url)

hsb['honcomp'] = np.where(hsb['write'] >= 60, 1, 0)
hsb['sex'] = np.where(hsb['female'] == ' female', 1, 0)

#m = rp.logistic("honcomp ~ sex + read + science", data=hsb)
m = rp.LogisticRegression("admit ~ gre + gpa + C(rank)", data=pol)

# %%
decimals={
    "Coef.": 2,
    "Std. Err.": 4,
    "test_stat": 4,
    "test_stat_p": 4,
    "CI": 2,
    "Root MSE": 4,
    "R-squared": 4,
    "Adj R-squared": 4,
    "Sum of Squares": 4,
    'Degrees of Freedom': 1,
    'Mean Squares': 4,
    'Effect size': 4
}
## Creating variable table information
regression_info = {m._DV_design_info.term_names[0]: [],
                        "Coef.": [],
                        "Std. Err.": [],
                        f"{m._test_stat_name}": [],
                        "p-value": [],
                        f"{int(m.CI_LEVEL * 100)}% Conf. Interval": []}

for column, beta, stderr, t, p, l_ci, u_ci in zip(m._IV_design_info.column_names,
                                                  m.model_data["betas"], m.model_data["standard_errors"],
                                                  m.model_data["test_stat"], m.model_data["test_stat_p_values"],
                                                  m.model_data["conf_int_lower"], m.model_data["conf_int_upper"]):

    print(column, beta, stderr, t, p, l_ci, u_ci)
    print(type(column), type(beta), type(stderr), type(t), type(p), type(l_ci), type(u_ci))

    regression_info[m._DV_design_info.term_names[0]].append(column)
    regression_info["Coef."].append(round(beta.item(), decimals["Coef."]))
    regression_info["Std. Err."].append(round(stderr.item(), decimals["Std. Err."]))
    regression_info[f"{m._test_stat_name}"].append(round(t.item(), decimals["test_stat"]))
    regression_info["p-value"].append(round(p.item(), decimals["test_stat_p"]))
    regression_info[f"{int(m.CI_LEVEL * 100)}% Conf. Interval"].append([round(l_ci.item(), decimals["CI"]),
                                                                                round(u_ci.item(), decimals["CI"])])

#self.regression_info = self._regression_base_table()


# %%
decimals={
                                     "Coef.": 2,
                                     "Std. Err.": 4,
                                     "test_stat": 4,
                                     "test_stat_p": 4,
                                     "CI": 2,
                                     "Root MSE": 4,
                                     "R-squared": 4,
                                     "Adj R-squared": 4,
                                     "Sum of Squares": 4,
                                     'Degrees of Freedom': 1,
                                     'Mean Squares': 4,
                                     'Effect size': 4
                                     }

regression_info = {m._DV_design_info.term_names[0]: [],
                   "Coef.": [],
                   "Std. Err.": [],
                   f"Z": [],
                   "p-value": [],
                   f"{int(m.CI_LEVEL * 100)}% Conf. Interval": []}

regression_info[m._DV_design_info.term_names[0]] = m._IV_design_info.column_names
regression_info["Coef."] = np.round(m.model_data["betas"].flatten(), decimals["Coef."]).tolist()
regression_info["Std. Err."] = np.round(m.model_data["standard_errors"].flatten(),
                                           decimals["Std. Err."]).tolist()
regression_info[f"Z"] = np.round(m.model_data["test_stat"].flatten(),
                                                         decimals["test_stat"]).tolist()
regression_info["p-value"] = np.round(m.model_data["test_stat_p_values"].flatten(),
                                         decimals["test_stat_p"]).tolist()
regression_info[f"{int(m.CI_LEVEL * 100)}% Conf. Interval"] = [list(x) for x in np.round(
    np.hstack((m.model_data["conf_int_lower"].flatten().reshape(-1, 1),
               m.model_data["conf_int_upper"].flatten().reshape(-1, 1))), decimals["CI"]).tolist()]



regression_info[m._DV_design_info.term_names[0]] = m._IV_design_info.column_names
regression_info["Coef."] = np.round(m.model_data["betas"].flatten(), decimals["Coef."]).tolist()
#regression_info["Coef."] = [np.round(val, decimals["Coef."]) for val in m.model_data["betas"]]



arr = np.array([1.234, 2.345, 3.456])
rounded_dict = {i: round(val, 2) for i, val in enumerate(arr)}



regression_info["Coef."].append(*np.round(m.model_data["betas"], decimals["Coef."]))
#regression_info["Coef."].append(*np.round(m.model_data["betas"], decimals["Coef."]))
regression_info["Std. Err."].append(np.round(m.model_data["standard_errors"], decimals["Std. Err."]))
regression_info[f"Z"].append(np.round(m.model_data["test_stat"], decimals["test_stat"]))
regression_info["p-value"].append(np.round(m.model_data["test_stat_p_values"], decimals["test_stat_p"]))
regression_info[f"{int(m.CI_LEVEL * 100)}% Conf. Interval"].append([np.round(m.model_data["conf_int_lower"], decimals["CI"]),
                                                                       np.round(m.model_data["conf_int_upper"], decimals["CI"])])

print(regression_info)

