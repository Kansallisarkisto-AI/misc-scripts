import optuna
from optuna.visualization.matplotlib import (
    plot_optimization_history,
    plot_intermediate_values,
    plot_param_importances,
)
import matplotlib.pyplot as plt

# Name of the Optuna study
STUDY_NAME = "study_name"
# Path for saving plots
SAVE_PATH = "./save_path/"

# Load the study from RDB storage
storage = optuna.storages.RDBStorage("sqlite:///optuna_trials.db")
study = optuna.load_study(study_name=STUDY_NAME, storage=storage)

# Plot optimization history
ax1 = plot_optimization_history(study)
plt.show()
ax1.figure.savefig(SAVE_PATH + "optimization_history.png", bbox_inches='tight')

# Plot intermediate values (if using pruning and intermediate reports)
ax2 = plot_intermediate_values(study)
plt.show()
ax2.figure.savefig(SAVE_PATH + "intermediate_values.png", bbox_inches='tight')

# Plot parameter importances
ax3 = plot_param_importances(study)
plt.show()
ax3.figure.savefig(SAVE_PATH + "param_importances.png", bbox_inches='tight')