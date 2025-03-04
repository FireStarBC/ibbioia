import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd

BASE_P50 = 26.0       # mmHg; baseline p50 at pH=7.4 and 37°C (Severinghaus, 1966)
BASE_HILL = 2.7       # Hill coefficient; (Dash & Bassingthwaighte, 2010)
R_Q = 0.85            # Respiratory quotient; (Lumb, 2016, p.123)
SOLUBILITY = 0.003    # mL O2/dL/mmHg; oxygen solubility (West's Respiratory Physiology)
METABOLIC_BASE = 3.5  # mL O2/kg/min; 1 MET = resting metabolic rate (Ganong, 2019)

# Temperature regulation constants (Gagge & Gonzalez, 1996, J Appl Physiol)
kT = 0.03   # °C/min per MET; rate of temperature increase due to metabolism
kCool = 0.1 # 1/min; cooling rate toward baseline (37°C)

# CO2 clearance rate (an estimated value)
kCO2 = 0.05  # 1/min

# Simplified Henderson–Hasselbalch parameters (literature approximation)
pH_base = 7.4
pH_slope = 0.0086  # per mmHg change in PCO2



class TransportModel:
    def __init__(self, weight=70, initial_po2=95, initial_pco2=40, initial_temp=37, noise=0.02):
        self.weight = weight  # in kg
        self.initial_po2 = initial_po2  # mmHg
        self.initial_pco2 = initial_pco2  # mmHg
        self.initial_temp = initial_temp  # °C
        self.noise = noise

        # For simplicity, assume resting metabolic rate = 1 MET.
        self.metabolic_rate = 1.0

    def p50(self, temp, pH):


        log_p50 = np.log10(BASE_P50) + 0.024 * (temp - 37) - 0.48 * (pH - pH_base)

        return 10 ** (log_p50 * np.random.normal(1, 0.02))

    def hill_coefficient(self):

        return max(2.0, BASE_HILL * np.random.normal(1, 0.05))

    def alveolar_ventilation(self, exercise_MET):
        ventilation_at_rest = 5
        return ventilation_at_rest * (1+ (0.6 * exercise_MET))
    def dissociation_curve(self, po2, temp, pH):
        n = self.hill_coefficient()
        p50_val = self.p50(temp, pH)
        return po2 ** n / (po2 ** n + p50_val ** n)

    def update_state(self, pco2, temp, dt, exercise_MET):

        # Calculate metabolic rate (1 MET at rest plus exercise additional METs)
        metabolic_rate = 1 + exercise_MET
        alveolar = self.alveolar_ventilation(exercise_MET)

        # Adjust the CO2 accumulation to have zero net production at rest
        dpco2 = dt * ((metabolic_rate - 1) * R_Q - kCO2 * alveolar * (pco2 - 40))
        dpco2 += (np.random.normal(0, 0.01))
        new_pco2 = pco2 + dpco2

        # Temperature change (zero net change at rest)
        dtemp = dt * (kT * (metabolic_rate - 1) - (kCool * (temp - 37) + (np.random.normal(0, self.noise))))
        new_temp = temp + dtemp

        # pH from simplified Henderson–Hasselbalch equation
        new_pH = pH_base - pH_slope * (new_pco2 - 40)

        # Oxygen partial pressure; a simple relationship with pCO2
        new_po2 = self.initial_po2 * (1 - 0.005 * (new_pco2 - 40))

        return new_pco2, new_temp, new_pH, new_po2

    def run_simulation(self, duration=60, dt=1, exercise_profile=lambda t: 0):

        times = np.arange(0, duration + dt, dt)
        # Lists to store simulation results
        pco2_vals = [self.initial_pco2]
        temp_vals = [self.initial_temp]
        pH_vals = [pH_base]  # starting at normal pH (7.4)
        po2_vals = [self.initial_po2]
        sat_vals = [self.dissociation_curve(self.initial_po2, self.initial_temp, pH_base)]

        for t in times[1:]:
            exercise_level = exercise_profile(t)
            new_pco2, new_temp, new_pH, new_po2 = self.update_state(pco2_vals[-1], temp_vals[-1], dt, exercise_level)
            pco2_vals.append(new_pco2)
            temp_vals.append(new_temp)
            pH_vals.append(new_pH)
            po2_vals.append(new_po2)
            sat_vals.append(self.dissociation_curve(new_po2, new_temp, new_pH))

        # Compile results into a DataFrame
        df = pd.DataFrame({
            'time (min)': times,
            'pCO2 (mmHg)': pco2_vals,
            'Temperature (°C)': temp_vals,
            'pH': pH_vals,
            'pO2 (mmHg)': po2_vals,
            'Saturation': sat_vals
        })
        return df


def run_multiple_simulations(num_trials=10, duration=60, dt=1, exercise_profile=lambda t: 10):
    all_results = None  # Placeholder for aggregated results

    for trial in range(1, num_trials + 1):
        model = TransportModel(noise=0.01)
        results = model.run_simulation(duration=duration, dt=dt, exercise_profile=exercise_profile)

        # Rename columns to include trial number
        results = results.rename(columns={col: f"{trial} {col}" for col in results.columns if col != 'time (min)'})

        if all_results is None:
            all_results = results  # First trial initializes the DataFrame
        else:
            all_results = all_results.merge(results.drop(columns=['time (min)']), left_index=True, right_index=True)

    return all_results


if __name__ == '__main__':
    # Run multiple simulations
    multi_trial_results = run_multiple_simulations(10)

    # Save to CSV
    multi_trial_results.to_csv('oxygen_transport_simulation_multi_trial.csv', index=False)

    # Plot results of the first trial as an example
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axs[0].plot(multi_trial_results['time (min)'], multi_trial_results['1 Saturation'], label='O2 Saturation')
    axs[0].set_ylabel('Saturation (%)')
    axs[0].legend()

    axs[1].plot(multi_trial_results['time (min)'], multi_trial_results['1 pCO2 (mmHg)'], 'r', label='pCO2')
    axs[1].set_ylabel('pCO2 (mmHg)')
    axs[1].legend()

    axs[2].plot(multi_trial_results['time (min)'], multi_trial_results['1 Temperature (°C)'], 'g', label='Temperature')
    axs[2].set_ylabel('Temperature (°C)')
    axs[2].set_xlabel('Time (min)')
    axs[2].legend()

    plt.tight_layout()
    plt.show()
