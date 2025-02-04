import pandas as pd
import matplotlib.pyplot as plt
from main import PublicTransitModel

# Load baseline data from CSV
baseline_data = pd.read_csv("datasets/daily_ridership.csv")
num_steps = len(baseline_data)

model_increase = PublicTransitModel(
    data_path="datasets/daily_ridership.csv",
    initial_revenue=0,
    initial_services=1,
    initial_convenience=1,
    fare=4.6,
    service_coeff=0.1,
    convenience_effect=0.2,
    travel_time_factor=0.05,
    awareness_coeff=0.05,
    employer_coeff=0.0,            
    ineffectiveness_base=0.2,
    ineffectiveness_sensitivity=4
)

model_no_pass = PublicTransitModel(
    data_path="datasets/daily_ridership.csv",
    initial_revenue=0,
    initial_services=1,
    initial_convenience=1,
    fare=4.6,
    service_coeff=0.1,
    convenience_effect=0.2,
    travel_time_factor=0.05,
    awareness_coeff=0.05,
    employer_coeff=0.0,           
    ineffectiveness_base=0.2,
    ineffectiveness_sensitivity=0.5
)

for _ in range(num_steps):
    model_increase.step()
    model_no_pass.step()

results_increase = model_increase.datacollector.get_model_vars_dataframe()
results_no_pass = model_no_pass.datacollector.get_model_vars_dataframe()

plt.figure(figsize=(12, 7))
plt.plot(baseline_data.index, baseline_data["Ridership"], label="Baseline Ridership (Input)", linewidth=2, color="black")
plt.plot(results_increase.index, results_increase["Ridership"], label="Dynamic Ridership (High Employer Pass Effect)", linewidth=2, color="green")
plt.plot(results_no_pass.index, results_no_pass["Ridership"], label="Dynamic Ridership (No Employer Pass Effect)", linewidth=2, color="red")
plt.xlabel("Day")
plt.ylabel("Ridership")
plt.title("Effect of Employer Transit Passes on Bus Ridership")
plt.legend()
plt.show()
