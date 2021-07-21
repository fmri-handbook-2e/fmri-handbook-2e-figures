# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# Figures from Chapter "Introduction" of Poldrack et al., Handbook of fMRI Data Analysis

# %% jupyter={"outputs_hidden": false}
import numpy as np
import pandas as pd
import sys
import os
import scipy.io
import git
import matplotlib.pyplot as plt

# use repo base directory as base dir
repo = git.Repo(os.path.dirname(__file__),
                search_parent_directories=True)
repo_path = repo.git.rev_parse("--show-toplevel")

# use repo path rather than pip installing the module
sys.path.append(repo_path)

from fmrihandbook.utils.config import get_config  # noqa: E402
from fmrihandbook.utils.figures import savefig  # noqa: E402
from fmrihandbook.utils.pubmed import get_pubmed_query_results  # noqa: E402

config = get_config(nbfile=__file__)
if not os.path.exists(config.figure_dir):
    os.makedirs(config.figure_dir)

# %% [markdown]
#
# # Figure 1: Pubmed hits for fMRI
# ## Get counts of pubmed hits by year

# %% jupyter={"outputs_hidden": false}

if not config.email:
    raise Exception('you must first set your email address for your Entrez account')

nhits = []
years = []
for year in range(1990, 2021):
    query = '("fMRI" OR "functional MRI" OR "functional magnetic resonance imaging") AND %d[DP]' % year
    results = get_pubmed_query_results(query, config.email)
    nhits.append(len(results['IdList']))
    years.append(year)


# %% [markdown]
# ## Plot hits by year

# %% jupyter={"outputs_hidden": false}

fig = plt.figure(figsize=config.figsize)
plt.plot(years, nhits)
plt.xlabel('Year', fontsize=18)
plt.ylabel('# of Pubmed abstracts', fontsize=18)

savefig(fig, 'pubmed_abstracts_by_year', config)

# %% [markdown]
# ## Plot cumulative hits

# %% jupyter={"outputs_hidden": false}

fig = plt.figure(figsize=config.figsize)
cumhits = np.zeros(len(years))
for i in range(len(years)):
    cumhits[i] = np.sum(nhits[:(i + 1)])
plt.plot(years, cumhits)
plt.xlabel('Year', fontsize=18)
plt.ylabel('Cumulative # of hits', fontsize=18)

savefig(fig, 'pubmed_abstracts_cumulative', config)

# %% [markdown]
# ## Save data to tsv file

# %% jupyter={"outputs_hidden": false}
hitsDf = pd.DataFrame({'years': years,
                       'hits': nhits,
                       'cumulativehits': cumhits})
hitsDf.to_csv(
    os.path.join(config.data_dir, 'pubmed_hits.tsv'),
    index=False, sep='\t')


# %% [markdown]
# # Figure 2: Hemodynamic response
# Using data provided by Stephen Engel

# %% jupyter={"outputs_hidden": false}

data = scipy.io.loadmat(os.path.join(config.data_dir, 'hrf_data.mat'))
hrfdata = data['allmnresps']
timepoints = np.arange(0, 16, 0.25)

fig = plt.figure(figsize=config.figsize)
plt.plot(timepoints, hrfdata[:, 1, :] * 100, linewidth=2)
plt.ylabel('% change in BOLD signal', fontsize=18)
plt.xlabel('Peristimulus time (secs)', fontsize=18)

savefig(fig, 'hrf', config)

# %%
