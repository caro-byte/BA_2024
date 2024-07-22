import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load the new dataset provided by the user
new_file_path_sales = 'Absatzzahlen_A_neu.csv'
new_df_sales = pd.read_csv(new_file_path_sales, sep=';', decimal=',', thousands='.', parse_dates=['Datum'], dayfirst=True)

# Daten nach Datum sortieren
new_df_sales = new_df_sales.sort_values('Datum')

# Rename the column for easier access
new_df_sales.rename(columns={'Summe von Verkaufte Einheiten': 'Absatzzahlen'}, inplace=True)

# Daten auf den Zeitraum ab 2018 filtern
new_df_sales_filtered = new_df_sales[new_df_sales['Datum'] >= '2018-01-01']

# Plot-Einstellungen
plt.figure(figsize=(13, 5))

# Plot the sales numbers on the left y-axis
plt.plot(new_df_sales_filtered['Datum'], new_df_sales_filtered['Absatzzahlen'], color='red', linewidth=1.5, linestyle='-')
plt.xlabel('Jahr', fontsize=14, fontname='Arial', fontweight='bold')
plt.ylabel('Abgestezte Units von A', fontsize=14, fontname='Arial', fontweight='bold', color='red')
plt.tick_params(axis='y', labelcolor='red')
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,.0f}".format(x).replace(',', '.')))
plt.gca().yaxis.set_major_locator(plt.MaxNLocator(nbins=6))  # Limit number of ticks on y-axis

# X-Achse formatieren
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # Yearly ticks

# Gitterlinien konfigurieren
plt.grid(False)

# Achsenbeschriftungen formatieren
plt.xticks(fontsize=12, fontname='Arial')
plt.gca().tick_params(axis='x', rotation=0)

# x-Achsenlimit setzen
plt.xlim(pd.Timestamp('2018-01-01'), max(new_df_sales_filtered['Datum']))

# Diagramm anzeigen
plt.tight_layout()
plt.show()
