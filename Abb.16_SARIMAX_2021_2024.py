import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import matplotlib.pyplot as plt
from itertools import product

# Load the data
df_absatzzahlen = pd.read_csv('Absatzzahlen_A_gefiltert.csv', sep=';', parse_dates=['Datum'], dayfirst=True)
df_marktanteile = pd.read_csv('Marktanteile_absolut.csv', sep=';', parse_dates=['Datum'], dayfirst=True)

# Filter data until April 2024
end_date = '2024-05-30'
df_absatzzahlen = df_absatzzahlen[df_absatzzahlen['Datum'] <= end_date]
df_marktanteile = df_marktanteile[df_marktanteile['Datum'] <= end_date]

# Calculate market share for A
df_marktanteile['Marktanteil_A'] = df_marktanteile['A'] / df_marktanteile['Summe Monat']

# Merge datasets
df = pd.merge(df_absatzzahlen, df_marktanteile[['Datum', 'Marktanteil_A']], on='Datum', how='inner')
df.columns = ['ds', 'y', 'Marktanteil_A']
df.set_index('ds', inplace=True)

# Interpolate missing values and drop any remaining NaNs
df['Marktanteil_A'].interpolate(method='linear', inplace=True)
df.dropna(inplace=True)

# Check data lengths
print(f"Length of dataset: {len(df)}")

# Define training, validation, and test period lengths based on available data
train_size = 28 # 15 months
val_size = 6 # 7 months (2.5 months approximately)
test_size = 6  # 7 months (2.5 months approximately)

# Ensure there is enough data for the splits
assert len(df) >= train_size + val_size + test_size, "Not enough data to split into train, validation, and test sets"

train_data = df.iloc[:train_size]
val_data = df.iloc[train_size:train_size + val_size]
test_data = df.iloc[train_size + val_size:train_size + val_size + test_size]

X_train, y_train = train_data['Marktanteil_A'], train_data['y']
X_val, y_val = val_data['Marktanteil_A'], val_data['y']
X_test, y_test = test_data['Marktanteil_A'], test_data['y']

# Verify data split lengths
print(f"Train length: {len(X_train)}, {len(y_train)}")
print(f"Validation length: {len(X_val)}, {len(y_val)}")
print(f"Test length: {len(X_test)}, {len(y_test)}")

# Ensure consistent lengths
assert len(X_train) == len(y_train), "Inconsistent lengths between X_train and y_train"
assert len(X_val) == len(y_val), "Inconsistent lengths between X_val and y_val"
assert len(X_test) == len(y_test), "Inconsistent lengths between X_test and y_test"

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

print(final_model_fit.summary())

# Make predictions on validation set
val_forecast = final_model_fit.predict(start=len(y_train), end=len(y_train) + len(y_val) - 1, exog=X_val)

# Update model with validation data
final_model_fit = final_model_fit.append(y_val, exog=X_val)

# Make predictions on test set
test_forecast = final_model_fit.predict(start=len(y_train) + len(y_val), end=len(y_train) + len(y_val) + len(y_test) - 1, exog=X_test)

# Function to calculate metrics
def calculate_metrics(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    rmse = sqrt(mse)
    r2 = r2_score(actual, predicted)
    return mse, rmse, r2

# Calculate metrics for train, validation, and test sets
train_pred = final_model_fit.fittedvalues[:len(y_train)]
val_pred = val_forecast
test_pred = test_forecast

train_mse, train_rmse, train_r2 = calculate_metrics(y_train, train_pred)
val_mse, val_rmse, val_r2 = calculate_metrics(y_val, val_pred)
test_mse, test_rmse, test_r2 = calculate_metrics(y_test, test_pred)

print(f"Train MSE: {train_mse}, Train RMSE: {train_rmse}, Train R2: {train_r2}")
print(f"Validation MSE: {val_mse}, Validation RMSE: {val_rmse}, Validation R2: {val_r2}")
print(f"Test MSE: {test_mse}, Test RMSE: {test_rmse}, Test R2: {test_r2}")

# Plotting
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
