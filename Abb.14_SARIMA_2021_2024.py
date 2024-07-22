import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from itertools import product

# Laden der Daten
df = pd.read_csv('Absatzzahlen_A_neu.csv', sep=';', parse_dates=['Datum'], dayfirst=True)
df.columns = ['ds', 'y']
df.set_index('ds', inplace=True)

# Filterung der Daten bis einschließlich April 2024
df = df[(df.index >= '2021-01-01') & (df.index <= '2024-05-01')]

# Splitten der Daten in Trainings-, Validierungs- und Testdaten
train_size = int(len(df) * 0.7)
val_size = int(len(df) * 0.15)
test_size = len(df) - train_size - val_size

train_data = df[:train_size]
val_data = df[train_size:train_size + val_size]
test_data = df[train_size + val_size:]

# Funktion für Grid Search
def grid_search_arima(data, p_values, d_values, q_values, P_values, D_values, Q_values, m):
    best_score, best_cfg = float("inf"), None
    for p, d, q, P, D, Q in product(p_values, d_values, q_values, P_values, D_values, Q_values):
        try:
            model = auto_arima(data, start_p=p, d=d, start_q=q, 
                               start_P=P, D=D, start_Q=Q, 
                               max_p=p, max_d=d, max_q=q, 
                               max_P=P, max_D=D, max_Q=Q, 
                               m=m, seasonal=True, trace=False,
                               error_action='ignore', suppress_warnings=True, stepwise=True)
            mse = mean_squared_error(data, model.predict_in_sample())
            if mse < best_score:
                best_score, best_cfg = mse, (p,d,q,P,D,Q)
        except:
            continue
    return best_score, best_cfg

# Parameter ranges für Grid Search
p_values = range(0, 3)
d_values = range(0, 2)
q_values = range(0, 3)
P_values = range(0, 2)
D_values = range(0, 2)
Q_values = range(0, 2)
m = 12  # Monatliche Saisonalität

# Grid Search durchführen
print("Performing Grid Search...")
best_score, best_cfg = grid_search_arima(train_data['y'], p_values, d_values, q_values, 
                                         P_values, D_values, Q_values, m)
print(f"Best ARIMA{best_cfg} MSE={best_score}")

# Bestes Modell trainieren
final_model = auto_arima(train_data['y'], start_p=best_cfg[0], d=best_cfg[1], start_q=best_cfg[2],
                         start_P=best_cfg[3], D=best_cfg[4], start_Q=best_cfg[5],
                         max_p=best_cfg[0], max_d=best_cfg[1], max_q=best_cfg[2],
                         max_P=best_cfg[3], max_D=best_cfg[4], max_Q=best_cfg[5],
                         m=m, seasonal=True, trace=False,
                         error_action='ignore', suppress_warnings=True, stepwise=True)

print(final_model.summary())

# Vorhersagen auf Validierungsdaten
val_forecast = final_model.predict(n_periods=len(val_data))
val_forecast = [max(0, x) for x in val_forecast]

# Aktualisierung des Modells mit Validierungsdaten
final_model.update(val_data['y'])

# Vorhersagen auf Testdaten
test_forecast = final_model.predict(n_periods=len(test_data))
test_forecast = [max(0, x) for x in test_forecast]

# Berechnung der Metriken für Trainings-, Validierungs- und Testdaten
def calculate_metrics(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    rmse = sqrt(mse)
    r2 = r2_score(actual, predicted)
    return mse, rmse, r2

train_pred = final_model.predict_in_sample(start=0, end=train_size-1)
train_mse, train_rmse, train_r2 = calculate_metrics(train_data['y'], train_pred)

val_mse, val_rmse, val_r2 = calculate_metrics(val_data['y'], val_forecast)
test_mse, test_rmse, test_r2 = calculate_metrics(test_data['y'], test_forecast)

print(f"Train MSE: {train_mse}, Train RMSE: {train_rmse}, Train R2: {train_r2}")
print(f"Validation MSE: {val_mse}, Validation RMSE: {val_rmse}, Validation R2: {val_r2}")
print(f"Test MSE: {test_mse}, Test RMSE: {test_rmse}, Test R2: {test_r2}")

# Plotting the validation and test data with predictions
fig, ax = plt.subplots(figsize=(18, 6))

# Plot historical data
ax.plot(df.index, df['y'], label='Trainingsdaten', color='black', linestyle='-')

# Plot validation data and forecast
ax.plot(val_data.index, val_data['y'], label='Validierungsdaten', color='blue', linestyle='-')
ax.plot(val_data.index, val_forecast, label='Validierungsvorhersage', color='blue', linestyle='--')

# Plot test data and forecast
ax.plot(test_data.index, test_data['y'], label='Testdaten', color='green', linestyle='-')
ax.plot(test_data.index, test_forecast, label='Testvorhersage', color='green', linestyle='--')

# Anpassung der Achsenbeschriftungen und der Legende
ax.set_xlabel('Monat', fontweight='bold', fontsize=20, fontname='Arial')
ax.set_ylabel('Abgesetzte Units von A', fontweight='bold', fontsize=20, fontname='Arial')

# Set the font size for the axis tick labels
ax.tick_params(axis='both', which='major', labelsize=18, labelcolor='black')

# Legende unterhalb des Graphen platzieren und anpassen
ax.legend(['Trainingsdaten', 'Validierungsdaten', 'Validierungsvorhersage', 'Testdaten', 'Testvorhersage'], 
          loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5, fontsize=18, frameon=False)

# Schritt 4: Graphen anzeigen mit Anpassungen
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
ax.axvline(pd.Timestamp('2023-05-01'), color='grey', linestyle=':', linewidth=1.7)
ax.axvline(pd.Timestamp('2023-11-01'), color='grey', linestyle=':', linewidth=1.7)

# Display the plot
plt.show()


