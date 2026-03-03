"""

In a command prompt (internal to Pychamr works also) activate the virtual environment for the
current version of researchpy and then install researchpy local version.

1. Activate virtual environment
    C:\Users\Servi\.virtualenvs\researchpy\v0.4.0\Scripts\activate

2. Install local version of researchpy in development mode
    python -m pip install -e .

"""
# %%
import os
import requests
import researchpy as rp
import pandas as pd
pd.set_option('display.max_columns',30,
              'display.width',1000)
import numpy as np

url = "https://stats.idre.ucla.edu/stat/stata/dae/binary.dta"
pol = pd.read_stata(url)

# This code block is for downloading the systolic.dta file from the Stata website and saving it locally. If the file
# already exists locally, it will read it directly from the provided local path.
systolic_local_path = "C:/Users/Corey Bryant/Downloads/systolic.dta"
if not os.path.exists(systolic_local_path):
    url = "https://www.stata-press.com/data/r19/systolic.dta"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        with open(systolic_local_path, "wb") as f:
            f.write(response.content)
    else:
        raise Exception(f"Failed to download file: {response.status_code}")

systolic = pd.read_stata(systolic_local_path)
del url, systolic_local_path, response

# %%
mdl = rp.model("admit ~ gre + gpa + C(rank)", data=pol)

# %%
m = rp.logistic("admit ~ gre + gpa + C(rank)", data=pol)
m.results()


# %% OLS regression
mols = rp.ols("systolic ~ C(drug) + C(disease) + C(drug):C(disease)", data=systolic)
desc, mod, table = mols.results()

print(desc, mod, table, sep="\n" * 2)


mols.IV.__dict__.values()

mols.IV.design_info


# %% ANOVA
mano = rp.anova("systolic ~ C(drug) + C(disease) + C(drug):C(disease)", data=systolic, sum_of_squares=3)
desc, table = mano.results()

print(desc, table, sep="\n" * 2)


# %% LM regression
mlm = rp.lm("systolic ~ C(drug) + C(disease) + C(drug):C(disease)", data=systolic)


desc, mod, table = mlm.results()

print(desc, mod, table, sep="\n" * 2)


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
