#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 13:15:27 2024

@author: carolinmagin
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 12:47:39 2024

@author: carolinmagin
"""

import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from datetime import timedelta

# Laden der Daten
df = pd.read_csv('Verschreibungsdaten_A.csv', sep=';', parse_dates=['Datum'], dayfirst=True)
df.columns = ['ds', 'y']
df.set_index('ds', inplace=True)

# Verwendung von auto_arima zur Bestimmung der optimalen Parameter
model = auto_arima(df['y'], seasonal=True, m=12,
                   start_p=1, start_q=1, max_p=3, max_q=3,
                   d=None, max_d=2, D=None, max_D=1,
                   trace=True, error_action='ignore', suppress_warnings=True,
                   stepwise=True)

print(model.summary())

# Schritt 1: Vorhersagen generieren
forecast, conf_int = model.predict(n_periods=12, return_conf_int=True, alpha=0.2)

# Adjusting negative forecast values to zero
forecast = [max(0, x) for x in forecast]
conf_int[:, 0] = [max(0, x) for x in conf_int[:, 0]]
conf_int[:, 1] = [max(0, x) for x in conf_int[:, 1]]

# Schritt 2: Zeitstempel für Vorhersagen erstellen
last_date = df.index[-1]
prediction_dates = [last_date + timedelta(days=30*x) for x in range(1, 13)]

# Schritt 3: Vorhersagedaten plotten
plt.figure(figsize=(15, 6))
plt.plot(df.index, df['y'], label='Historische Daten', color='black', linestyle='-')
plt.plot(prediction_dates, forecast, label='Prognostizierte Daten', color='red', linestyle='--')

# Plotting the confidence interval
plt.fill_between(prediction_dates, conf_int[:, 0], conf_int[:, 1], color='red', alpha=0.2, label='80% Konfidenzintervall')

plt.title('Vorhersage der nächsten 12 Monate (basierend auf 6 Jahren historischer Daten)')
plt.xlabel('Jahr', fontweight='bold')
plt.ylabel('Verschreibungen Einheiten A', fontweight='bold')

# Legende oben links platzieren
plt.legend(loc='upper left')

# Set x-axis limits to start in 2018 and end in 2025
plt.xlim(pd.Timestamp('2018-01-01'), pd.Timestamp('2025-01-15'))

# Schritt 4: Graphen anzeigen mit Anpassungen
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
plt.axvline(pd.Timestamp('2024-02-01'), color='red', linestyle='-', linewidth=2)
plt.show()