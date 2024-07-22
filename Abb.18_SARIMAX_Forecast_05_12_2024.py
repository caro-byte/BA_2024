#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:59:07 2024

@author: carolinmagin
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:43:44 2024

@author: carolinmagin
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from itertools import product

# Load the data
df_absatzzahlen = pd.read_csv('Absatzzahlen_A_gefiltert.csv', sep=';', parse_dates=['Datum'], dayfirst=True)
df_marktanteile = pd.read_csv('Marktanteile_absolut.csv', sep=';', parse_dates=['Datum'], dayfirst=True)

# Filter data from January 2021 to April 2024
start_date = '2021-01-01'
end_date = '2024-04-30'
df_absatzzahlen = df_absatzzahlen[(df_absatzzahlen['Datum'] >= start_date) & (df_absatzzahlen['Datum'] <= end_date)]
df_marktanteile = df_marktanteile[(df_marktanteile['Datum'] >= start_date) & (df_marktanteile['Datum'] <= end_date)]

# Calculate market share for A
df_marktanteile['Marktanteil_A'] = df_marktanteile['A'] / df_marktanteile['Summe Monat']

# Merge datasets
df = pd.merge(df_absatzzahlen, df_marktanteile[['Datum', 'Marktanteil_A']], on='Datum', how='inner')
df.columns = ['ds', 'y', 'Marktanteil_A']
df.set_index('ds', inplace=True)

# Interpolate missing values and drop any remaining NaNs
df['Marktanteil_A'].interpolate(method='linear', inplace=True)
df.dropna(inplace=True)

# Use all available data for training
X_train, y_train = df['Marktanteil_A'], df['y']

# Grid search function for SARIMA parameters
def grid_search_sarima(endog, exog, p_values, d_values, q_values, P_values, D_values, Q_values, m):
    best_score, best_cfg = float("inf"), None
    for p, d, q, P, D, Q in product(p_values, d_values, q_values, P_values, D_values, Q_values):
        try:
            model = sm.tsa.SARIMAX(endog, exog=exog, order=(p, d, q), seasonal_order=(P, D, Q, m))
            model_fit = model.fit(disp=False)
            mse = mean_squared_error(endog, model_fit.fittedvalues)
            if mse < best_score:
                best_score, best_cfg = mse, (p, d, q, P, D, Q)
        except:
            continue
    return best_score, best_cfg

# Define parameter ranges for grid search
p_values = range(0, 3)
d_values = range(0, 2)
q_values = range(0, 3)
P_values = range(0, 2)
D_values = range(0, 2)
Q_values = range(0, 2)
m = 12  # Monthly seasonality

# Perform grid search
print("Performing Grid Search...")
best_score, best_cfg = grid_search_sarima(y_train, X_train, p_values, d_values, q_values, P_values, D_values, Q_values, m)
print(f"Best SARIMA{best_cfg} MSE={best_score}")

# Train the best model
final_model = sm.tsa.SARIMAX(y_train, exog=X_train, order=(best_cfg[0], best_cfg[1], best_cfg[2]),
                             seasonal_order=(best_cfg[3], best_cfg[4], best_cfg[5], m))
final_model_fit = final_model.fit(disp=False)

# Create future dates for prediction (May to December 2024)
future_dates = pd.date_range(start='2024-05-01', end='2024-12-31', freq='M')
future_exog = pd.DataFrame(index=future_dates, columns=['Marktanteil_A'])
future_exog['Marktanteil_A'] = X_train.iloc[-1]  # Use the last known market share for future predictions

# Make predictions for May to December 2024
forecast = final_model_fit.get_forecast(steps=len(future_dates), exog=future_exog)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int(alpha=0.2)  # 80% confidence interval

# Plotting
fig, ax = plt.subplots(figsize=(16, 8))  # Increased width

# Set font and font size
plt.rcParams.update({'font.size': 14, 'font.family': 'Arial'})

# Plot historical data
ax.plot(df.index, df['y'], label='Historische Daten', color='#3E207D', linewidth=2)

# Plot forecast
ax.plot(future_dates, forecast_mean, label='Vorhersage 2024', color='black', linestyle=':', linewidth=2)

# Plot confidence intervals
ax.fill_between(future_dates, forecast_ci.iloc[:, 0], forecast_mean, color='#DB003E', alpha=0.6, label='Worst-Case')
ax.fill_between(future_dates, forecast_mean, forecast_ci.iloc[:, 1], color='#96C640', alpha=0.6, label='Best-Case')

# Remove vertical grid lines
ax.grid(axis='y', linestyle='--', color='grey', linewidth=0.5)

# Add labels
ax.set_xlabel('Monat', fontweight='bold', fontsize=16, fontname='Arial')
ax.set_ylabel('Abgesetzte Units von A', fontweight='bold', fontsize=16, fontname='Arial')

# Format x-axis for quarters
def month_to_quarter(month):
    return (month - 1) // 3 + 1

def custom_formatter(x, pos=None):
    date = mdates.num2date(x)
    quarter = month_to_quarter(date.month)
    if quarter == 1:  # Nur für das erste Quartal das Jahr anzeigen
        return f'{date.year} Q{quarter}'
    else:
        return f'Q{quarter}'

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(plt.FuncFormatter(custom_formatter))

# Format y-axis for thousands with separator
def thousands_formatter(x, pos):
    return f'{x:,.0f}'.replace(',', '.')

ax.yaxis.set_major_formatter(ticker.FuncFormatter(thousands_formatter))

# Adjust axis labels
plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center', fontsize=14, fontname='Arial')
plt.setp(ax.yaxis.get_majorticklabels(), fontsize=14, fontname='Arial')

# Add vertical dashed lines
line_1 = pd.Timestamp('2024-05-01')
line_2 = pd.Timestamp('2025-01-01')
ax.axvline(line_1, color='grey', linestyle=':', linewidth=1.7)
ax.axvline(line_2, color='grey', linestyle=':', linewidth=1.7)

# Annotate the vertical lines
ax.annotate('05-2024', xy=(line_1, ax.get_ylim()[1]), xytext=(line_1, ax.get_ylim()[1] * 0.95),
            arrowprops=dict(facecolor='grey', shrink=0.05), fontsize=14, fontname='Arial', ha='center')
ax.annotate('01-2025', xy=(line_2, ax.get_ylim()[1]), xytext=(line_2, ax.get_ylim()[1] * 0.95),
            arrowprops=dict(facecolor='grey', shrink=0.05), fontsize=14, fontname='Arial', ha='center')

# Extend the historical data to include the forecasted values
combined_dates = pd.concat([pd.Series(df.index), pd.Series(future_dates)])
combined_values = pd.concat([df['y'], forecast_mean])

# Plot the combined data
ax.plot(combined_dates, combined_values, color='black', linestyle=':', linewidth=2)

# Place legend in one line below the graph
handles, labels = ax.get_legend_handles_labels()
legend = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                   ncol=len(labels), frameon=False, fontsize=14)

# Show plot
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)  # Adjusted to accommodate the legend
plt.show()

# Print forecast values for May to December 2024
print("Vorhersage für Mai bis Dezember 2024:")
print(pd.DataFrame({
    'ds': future_dates,
    'yhat': forecast_mean,
    'yhat_lower': forecast_ci.iloc[:, 0],
    'yhat_upper': forecast_ci.iloc[:, 1]
}))