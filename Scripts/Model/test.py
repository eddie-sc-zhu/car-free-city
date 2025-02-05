import pandas as pd
import matplotlib.pyplot as plt
from main import PublicTransitModel

baseline_data = pd.read_csv("datasets/daily_ridership.csv")
num_steps = len(baseline_data)

model_increase = PublicTransitModel(intitial_employer=0, employer_coeff=0.01)
model_no_pass = PublicTransitModel(employer_coeff=0.0)

for _ in range(num_steps):
    model_increase.step()
    model_no_pass.step()

results_increase = model_increase.datacollector.get_model_vars_dataframe()
results_no_pass = model_no_pass.datacollector.get_model_vars_dataframe()

min_length = min(len(baseline_data), len(results_increase), len(results_no_pass))
baseline_truncated = baseline_data.iloc[:min_length]
results_increase_truncated = results_increase.iloc[:min_length]
results_no_pass_truncated = results_no_pass.iloc[:min_length]

diff_increase = results_increase_truncated["Ridership"].values - baseline_truncated["Ridership"].values
diff_no_pass = results_no_pass_truncated["Ridership"].values - baseline_truncated["Ridership"].values

plt.plot(baseline_truncated.index, baseline_truncated["Ridership"], label="Baseline Ridership (Input)", linewidth=2, color="black")
plt.plot(results_increase_truncated.index, results_increase_truncated["Ridership"], label="Dynamic Ridership (High Employer Pass Effect)", linewidth=2, color="green")
plt.plot(results_no_pass_truncated.index, results_no_pass_truncated["Ridership"], label="Dynamic Ridership (No Employer Pass Effect)", linewidth=2, color="red")
plt.xlabel("Day")
plt.ylabel("Ridership")
plt.title("Effect of Employer Transit Passes on Bus Ridership")
plt.legend()
plt.show()

plt.plot(baseline_truncated.index, diff_increase, label="Difference (High Employer Pass Effect)", linewidth=2, color="green")
plt.plot(baseline_truncated.index, diff_no_pass, label="Difference (No Employer Pass Effect)", linewidth=2, color="red")
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.xlabel("Day")
plt.ylabel("Difference in Ridership")
plt.title("Deviation of Modeled Ridership from Baseline")
plt.legend()
plt.show()
