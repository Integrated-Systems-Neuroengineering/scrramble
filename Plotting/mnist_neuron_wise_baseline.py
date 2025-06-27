"""
Plotting MNIST baselines.
Here we consider ScRRAMBLe against feedforward networks with the same number of NEURONS.
"""

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from datetime import date

import os
import pickle

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{type1cm}'

DATA_PATH = "/Users/vikrantjaltare/OneDrive - UC San Diego/scrramble/data"

today = date.today().isoformat()

ff_data_path = os.path.join(DATA_PATH, "baseline_ff_cores_par_2M_a_97_2025-05-20.pkl")
# log_data_path = os.path.join(DATA_PATH, "baseline_logistic_regression_a_92_2025-05-13.pkl")

ff_data = pd.DataFrame(pickle.load(open(ff_data_path, "rb")))
print(ff_data.head())
# log_data = pd.read_pickle(log_data_path)

ff_best_accuracy = ff_data["test_accuracy"].max()
# log_best_accuracy = log_data["test_accuracy"].max()
ff_min_loss = ff_data["test_loss"].min()
# log_min_loss = log_data["test_loss"].min()

# print all 4
print(f"FF best accuracy: {ff_best_accuracy}")
# print(f"Logistic best accuracy: {log_best_accuracy}")
print(f"FF min loss: {ff_min_loss}")
# print(f"Logistic min loss: {log_min_loss}")

# load the dataframe for 32 cores
filepath = os.path.join(DATA_PATH, "mnist_arch_dict_cores_32_resmp_10_2025-05-20.pkl")
ws16_df = pd.DataFrame(pickle.load(open(filepath, "rb")))

# line plots
pal1 = sns.color_palette("Set2", len(ws16_df['arch'].unique()))

fig, ax = plt.subplots(1, 2, dpi=110, figsize=(7, 3.5))

# for tick in ax[0].get_xticklabels(which='both'):
#     tick.set_fontname('Arial')

# ws16_df = ws16_df.loc[ws16_df['arch'] < 1]

sns.lineplot(data=ws16_df, x='avg_slot_connectivity', y='train_accuracy', err_style='band', ax=ax[0], color=pal1[0], label='ScRRAMBLe Train accuracy', lw=1.5, alpha=.7, marker='o', markersize=5)
sns.lineplot(data=ws16_df, x='avg_slot_connectivity', y='test_accuracy', err_style='band', ax=ax[0], color=pal1[1], label='ScRRAMBLe Test accuracy', lw=1.5, alpha=.7, marker='o', markersize=5)
ax[0].set_xlabel(r"Average slot connectivity $\lambda/N_c$", fontsize = 14)
ax[0].set_ylabel(r"Accuracy", fontsize = 14)
ax[0].set_xticks(np.arange(0, 11, 2))
ax[0].set_xticklabels([f"{x/(20*4):.2f}" for x in np.arange(0, 11, 2)])
ax[0].axhline(y=ff_best_accuracy, color='0.3', linestyle='--', label='feedforward network', lw = 2.5, alpha=.7)
# ax[0].axhline(y=log_best_accuracy, color='0.6', linestyle='-.', label='logistic regressor', lw = 2.5, alpha=.7)
ax[0].minorticks_on()
ax[0].legend([], [], frameon=False)
ax[0].set_ylim(0.7, 1.0)
ax[0].set_xlim(0, 10)

sns.lineplot(data=ws16_df, x='avg_slot_connectivity', y='train_loss', err_style='band', ax=ax[1], label="ScRRAMBLe Train loss", color=pal1[0], lw=1.5, alpha=.7, marker='o', markersize=5)
sns.lineplot(data=ws16_df, x='avg_slot_connectivity', y='test_loss', err_style='band', ax=ax[1], label="ScRRAMBLe Test loss", color=pal1[1], lw=1.5, alpha=.7, marker='o', markersize=5)
ax[1].axhline(y=ff_min_loss, color='0.3', linestyle='--', label='feedforward network', lw = 2.5, alpha=.7)
# ax[1].axhline(y=log_min_loss, color='0.6', linestyle='-.', label='logistic regressor', lw = 2.5, alpha=.7)
ax[1].minorticks_on()
ax[1].legend(frameon=False)
ax[1].set_xlabel(r"Average slot connectivity $\lambda/N_c$", fontsize = 14)
ax[1].set_ylabel(r"Loss", fontsize = 14)
ax[1].set_xticks(np.arange(0, 11, 2))
ax[1].set_xticklabels([f"{x/(20*4):.2f}" for x in np.arange(0, 11, 2)])
ax[1].set_ylim(1.45, 1.7)
ax[1].set_xlim(0, 10)

plt.tight_layout()
sns.despine()
plt.savefig(f"plots/mnist_parameter_wise_baseline_cores_32_{today}.pdf", bbox_inches="tight", dpi=600, transparent=True)
plt.show()  