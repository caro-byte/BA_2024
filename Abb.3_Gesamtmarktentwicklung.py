#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 12:04:21 2024

@author: carolinmagin
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# CSV-Datei einlesen
file_path = 'Marktanteile_absolut.csv'
df = pd.read_csv(file_path, sep=';', decimal=',', thousands='.', parse_dates=['Datum'], dayfirst=True)

# Daten nach Datum sortieren
df = df.sort_values('Datum')

# Daten auf den Zeitraum von 01-2021 bis 04-2024 filtern
start_date = '2021-01-01'
end_date = '2024-04-30'
df = df[(df['Datum'] >= start_date) & (df['Datum'] <= end_date)]

# Plot-Einstellungen
plt.figure(figsize=(13, 5))

# Plot the total market development
plt.plot(df['Datum'], df['Summe Monat'], color='black', linewidth=1.5, linestyle='-')

# Add trend line
z = np.polyfit(df['Datum'].map(mdates.date2num), df['Summe Monat'], 1)
p = np.poly1d(z)
plt.plot(df['Datum'], p(mdates.date2num(df['Datum'])), color='gray', linestyle='--')

# Label placement between Q2 and Q3 2021
label_date = pd.Timestamp('2021-07-01')
label_value = df.loc[df['Datum'] == label_date, 'Summe Monat'].values[0]


# Diagramm anpassen
plt.xlabel('Monat', fontsize=14, fontname='Arial', fontweight='bold')
plt.ylabel('Verschreibungszahlen in EQs', fontsize=14, fontname='Arial', fontweight='bold')

# X-Achse formatieren
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Quartalsweise Ticks

# Y-Achse formatieren (Tausendertrennzeichen mit Punkt)
plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,.0f}".format(x).replace(',', '.')))
plt.gca().yaxis.set_major_locator(plt.MaxNLocator(nbins=6))  # Limit number of ticks on y-axis

# Gitterlinien konfigurieren
plt.grid(False)

# Benutzerdefinierte Funktionen für Quartalsanzeige
def month_to_quarter(month):
    return (month - 1) // 3 + 1

def custom_formatter(x, pos=None):
    date = mdates.num2date(x)
    quarter = month_to_quarter(date.month)
    if quarter == 1:  # Nur für das erste Quartal das Jahr anzeigen
        return f'{date.year} Q{quarter}'
    else:
        return f'Q{quarter}'

# Haupt-Ticks der x-Achse auf Quartalsbasis setzen
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(custom_formatter))

# Achsenbeschriftungen formatieren
plt.xticks(fontsize=12, fontname='Arial')
plt.yticks(fontsize=12, fontname='Arial')

# x-Achsenlimit setzen
plt.xlim(min(df['Datum']), max(df['Datum']))

# Diagramm anzeigen
plt.tight_layout()
plt.show()
