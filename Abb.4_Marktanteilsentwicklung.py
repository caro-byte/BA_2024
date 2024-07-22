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

# Convert absolute numbers to market shares
total = df[['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']].sum(axis=1)
df_market_share = df.copy()
for column in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']:
    df_market_share[column] = df[column] / total * 100

# Plot-Einstellungen
plt.figure(figsize=(13, 6)) #dick

# Stildefinitionen für die Linien
style_dict = {
    'A': {'color': 'blue', 'linewidth': 3, 'linestyle': '-'},
    'B': {'color': 'black', 'linewidth': 3, 'linestyle': '--'},
    'C': {'color': 'black', 'linewidth': 1, 'linestyle': '-'},
    'D': {'color': 'black', 'linewidth': 1, 'linestyle': '--'},
    'E': {'color': 'black', 'linewidth': 1, 'linestyle': '-'},
    'F': {'color': 'black', 'linewidth': 2, 'linestyle': '-'},
    'G': {'color': 'black', 'linewidth': 1, 'linestyle': '-'},
    'H': {'color': 'black', 'linewidth': 1, 'linestyle': ':'},
    'I': {'color': 'black', 'linewidth': 1, 'linestyle': '-'},
}

# Linien für jeden Marktanteil zeichnen
for column in df_market_share.columns:
    if column in style_dict:
        style = style_dict[column]
        line = plt.plot(df_market_share['Datum'], df_market_share[column], **style)[0]
        
        # Beschriftung zwischen Q2 und Q3 2021
        label_date = pd.Timestamp('2021-07-01')
        label_value = df_market_share.loc[df_market_share['Datum'] == label_date, column].values[0]
        
        # Beschriftung platzieren
        plt.text(label_date, label_value, column, fontsize=12, fontname='Arial', 
                 va='center', ha='center', color=style['color'], 
                 bbox=dict(facecolor='white', edgecolor='none'))

# Diagramm anpassen
plt.xlabel('Monat', fontsize=14, fontname='Arial', fontweight='bold')
plt.ylabel('Marktanteile am FSH-HMG-Gesamtmarkt (%)', fontsize=14, fontname='Arial', fontweight='bold')

# X-Achse formatieren
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Quartalsweise Ticks

# Y-Achse formatieren
plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,.1f}".format(x)))

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
plt.xlim(min(df_market_share['Datum']), max(df_market_share['Datum']))

# Diagramm anzeigen
plt.tight_layout()
plt.show()
