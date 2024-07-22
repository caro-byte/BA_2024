import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

# Load the datasets
absatzzahlen_df = pd.read_csv('Absatzzahlen_A_neu.csv', delimiter=';')
marktanteile_df = pd.read_csv('Marktanteile_absolut.csv', delimiter=';')

# Parse the dates
absatzzahlen_df['Datum'] = pd.to_datetime(absatzzahlen_df['Datum'], format='%d.%m.%y')
marktanteile_df['Datum'] = pd.to_datetime(marktanteile_df['Datum'], format='%d.%m.%y')

# Filter the data for the years 2021 to 2024
absatzzahlen_filtered = absatzzahlen_df[(absatzzahlen_df['Datum'].dt.year >= 2021) & (absatzzahlen_df['Datum'].dt.year <= 2024)]
marktanteile_filtered = marktanteile_df[(marktanteile_df['Datum'].dt.year >= 2021) & (marktanteile_df['Datum'].dt.year <= 2024)]

# Calculate moving averages
absatzzahlen_filtered['Gleitender Durchschnitt'] = absatzzahlen_filtered['Summe von Verkaufte Einheiten'].rolling(window=3).mean()
marktanteile_filtered['Marktanteil A'] = marktanteile_filtered['A'] / marktanteile_filtered['Summe Monat'] * 100
marktanteile_filtered['Gleitender Durchschnitt'] = marktanteile_filtered['Marktanteil A'].rolling(window=3).mean()

# Plotting with new requirements
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot Absatzzahlen
color = 'red'
ax1.set_xlabel('Jahr', fontsize=14, fontweight='bold', fontname='Arial')
ax1.set_ylabel('Abgesetzte Units von A', color=color, fontsize=14, fontweight='bold', fontname='Arial')
ax1.plot(absatzzahlen_filtered['Datum'], absatzzahlen_filtered['Summe von Verkaufte Einheiten'], color=color, label='Absatzzahlen')
ax1.plot(absatzzahlen_filtered['Datum'], absatzzahlen_filtered['Gleitender Durchschnitt'], color=color, linestyle='--', label='Gleitender Durchschnitt Absatzzahlen')
ax1.tick_params(axis='y', labelcolor=color, labelsize=12, labelrotation=0, direction='inout', length=5, width=2)
ax1.tick_params(axis='x', labelsize=12, labelrotation=0, direction='inout', length=5, width=2)

# Format y-axis with thousand separators
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x).replace(',', '.')))

# Plot Marktanteile on the same graph with a different y-axis
ax2 = ax1.twinx()
color = 'black'
ax2.set_ylabel('Marktanteil A (%)', color=color, fontsize=14, fontweight='bold', fontname='Arial')
ax2.plot(marktanteile_filtered['Datum'], marktanteile_filtered['Marktanteil A'], color=color, label='Marktanteil A')
ax2.plot(marktanteile_filtered['Datum'], marktanteile_filtered['Gleitender Durchschnitt'], color=color, linestyle='--', label='Gleitender Durchschnitt Marktanteil A')
ax2.tick_params(axis='y', labelcolor=color, labelsize=12, labelrotation=0, direction='inout', length=5, width=2)
ax2.spines['right'].set_visible(True)
ax2.set_ylim(0, max(marktanteile_filtered['Marktanteil A'].max(), marktanteile_filtered['Gleitender Durchschnitt'].max()) * 1.1)

# Set major ticks format to years
ax1.xaxis.set_major_locator(mdates.YearLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Remove gridlines
ax1.grid(False)
ax2.grid(False)

# Adjust x-axis limits to reduce spacing
ax1.set_xlim([absatzzahlen_filtered['Datum'].min(), absatzzahlen_filtered['Datum'].max()])
ax1.set_ylim(0, max(absatzzahlen_filtered['Summe von Verkaufte Einheiten'].max(), absatzzahlen_filtered['Gleitender Durchschnitt'].max()) * 1.1)

# Adding legends below the plots with increased font size
fig.tight_layout(rect=[0, 0.1, 1, 0.95])
ax1.legend(loc='upper left', bbox_to_anchor=(0, -0.15), frameon=False, fontsize=14)
ax2.legend(loc='upper right', bbox_to_anchor=(1, -0.15), frameon=False, fontsize=14)

plt.show()
