#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 12:15:15 2024

@author: carolinmagin
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# CSV-Dateien einlesen
file_path = 'Marktanteile_absolut.csv'
df = pd.read_csv(file_path, sep=';', decimal=',', thousands='.', parse_dates=['Datum'], dayfirst=True)

file_path_sales = 'Absatzzahlen_A_neu.csv'
df_sales = pd.read_csv(file_path_sales, sep=';', decimal=',', thousands='.', parse_dates=['Datum'], dayfirst=True)

# Daten nach Datum sortieren
df = df.sort_values('Datum')
df_sales = df_sales.sort_values('Datum')

# Daten auf den Zeitraum von 01-2021 bis 04-2024 filtern
start_date = '2021-01-01'
end_date = '2024-04-30'
df = df[(df['Datum'] >= start_date) & (df['Datum'] <= end_date)]
df_sales = df_sales[(df_sales['Datum'] >= start_date) & (df_sales['Datum'] <= end_date)]

# Rename the column for easier access
df_sales.rename(columns={'Summe von Verkaufte Einheiten': 'Absatzzahlen'}, inplace=True)



# Plot-Einstellungen
fig, ax1 = plt.subplots(figsize=(13, 5))

# Plot the sales numbers on the left y-axis
line1, = ax1.plot(df_sales['Datum'], df_sales['Absatzzahlen'], color='red', linewidth=1.5, linestyle='-', label='Absatzzahlen')
ax1.set_xlabel('Monat', fontsize=14, fontname='Arial', fontweight='bold')
ax1.set_ylabel('Absatzzahlen', fontsize=14, fontname='Arial', fontweight='bold', color='red')
ax1.tick_params(axis='y', labelcolor='red')
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,.0f}".format(x).replace(',', '.')))
ax1.yaxis.set_major_locator(plt.MaxNLocator(nbins=6))  # Limit number of ticks on y-axis

# Add moving average for sales numbers
sales_moving_avg = df_sales['Absatzzahlen'].rolling(window=3).mean()
line2, = ax1.plot(df_sales['Datum'], sales_moving_avg, color='red', linestyle='--', linewidth=1.5, label='Absatzzahlen (gleitender Durchschnitt)')

# Create a second y-axis for the total market development
ax2 = ax1.twinx()
line3, = ax2.plot(df['Datum'], df['Summe Monat'], color='black', linewidth=1.5, linestyle='-', label='Verschreibungszahlen in EQs')
ax2.set_ylabel('Verschreibungszahlen in EQs', fontsize=14, fontname='Arial', fontweight='bold', color='black')
ax2.tick_params(axis='y', labelcolor='black')
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,.0f}".format(x).replace(',', '.')))
ax2.yaxis.set_major_locator(plt.MaxNLocator(nbins=6))  # Limit number of ticks on y-axis

# Add moving average for total market development
market_moving_avg = df['Summe Monat'].rolling(window=3).mean()
line4, = ax2.plot(df['Datum'], market_moving_avg, color='black', linestyle='--', linewidth=1.5, label='Verschreibungszahlen in EQs (gleitender Durchschnitt)')

# X-Achse formatieren
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Quartalsweise Ticks

# Gitterlinien konfigurieren
ax1.grid(False)
ax2.grid(False)

# Benutzerdefinierte Funktionen für Quartalsanzeige
def custom_formatter(x, pos=None):
    date = mdates.num2date(x)
    quarter = (date.month - 1) // 3 + 1
    if quarter == 1:  # Nur für das erste Quartal das Jahr anzeigen
        return f'{date.year} Q{quarter}'
    else:
        return f'Q{quarter}'

# Haupt-Ticks der x-Achse auf Quartalsbasis setzen
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax1.xaxis.set_major_formatter(plt.FuncFormatter(custom_formatter))

# Achsenbeschriftungen formatieren
plt.xticks(fontsize=12, fontname='Arial')
ax1.tick_params(axis='x', rotation=0)
ax2.tick_params(axis='x', rotation=0)

# x-Achsenlimit setzen
plt.xlim(min(df['Datum']), max(df['Datum']))

# Add legend
lines = [line1, line2, line3, line4]
labels = [line.get_label() for line in lines]
plt.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=14, frameon=False, prop={'family': 'Arial'})

# Diagramm anzeigen
plt.tight_layout()
plt.show()
