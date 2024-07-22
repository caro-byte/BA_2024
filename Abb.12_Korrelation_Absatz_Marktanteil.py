import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV files
file_path_market_share = 'Marktanteile_absolut.csv'
file_path_sales = 'Absatzzahlen_A_neu.csv'

data_market_share = pd.read_csv(file_path_market_share, delimiter=';')
data_sales = pd.read_csv(file_path_sales, delimiter=';')

# Converting the 'Datum' column to datetime format in both datasets
data_market_share['Datum'] = pd.to_datetime(data_market_share['Datum'], format='%d.%m.%y')
data_sales['Datum'] = pd.to_datetime(data_sales['Datum'], format='%d.%m.%y')

# Merge the datasets on the 'Datum' column
merged_data = pd.merge(data_sales, data_market_share, on='Datum', how='inner')

# Calculate the market share of product A
merged_data['Marktanteil von A'] = merged_data['A'] / merged_data['Summe Monat']

# Creating the scatter plot with the specified adjustments
plt.figure(figsize=(10.5, 5))
plt.scatter(merged_data['Summe von Verkaufte Einheiten'], merged_data['Marktanteil von A'], color='black', marker='o', label='Datenpunkte')
plt.xlabel('Abgesetzte Units von A', fontsize=14, fontweight='bold', fontname='Arial')
plt.ylabel('Marktanteil von A', fontsize=14, fontweight='bold', fontname='Arial')
plt.xlim(merged_data['Summe von Verkaufte Einheiten'].min(), merged_data['Summe von Verkaufte Einheiten'].max())
plt.ylim(0.15, max(0.5, merged_data['Marktanteil von A'].max() + 0.05))  # Start from 15%, go up to at least 50% or higher if needed

# Set y-axis ticks to 5% steps, starting from 15%
y_ticks = np.arange(0.15, plt.ylim()[1], 0.05)
plt.yticks(y_ticks)

# Adding a dotted grey trend line
z = np.polyfit(merged_data['Summe von Verkaufte Einheiten'], merged_data['Marktanteil von A'], 1)
p = np.poly1d(z)
x_values = merged_data['Summe von Verkaufte Einheiten']
plt.plot(x_values, p(x_values), linestyle=':', color='grey')

# Adding text label for the trend line
trend_x_position = (x_values.max() + x_values.min()) / 2
trend_y_position = p(trend_x_position)
plt.text(trend_x_position, trend_y_position, 'Trend', fontsize=12, color='grey', ha='left')

# Remove grid lines
plt.grid(False)

plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x)).replace(",", ".")))
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, loc: "{:.0%}".format(y)))

plt.show()

# Calculating the correlation coefficient between the two variables
correlation_coefficient = merged_data['Summe von Verkaufte Einheiten'].corr(merged_data['Marktanteil von A'])

print(f"Correlation coefficient: {correlation_coefficient:.4f}")