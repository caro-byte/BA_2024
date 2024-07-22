import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = 'eu_statistics_age_mom.csv'
data = pd.read_csv(file_path, delimiter=';')

# Converting the 'Alter' column to numeric, replacing the comma with a dot
data['Alter'] = data['Alter'].str.replace(',', '.').astype(float)

# Converting the 'Alter' column back to string with comma for displaying purposes
data['Alter'] = data['Alter'].astype(str).str.replace('.', ',')

# Plotting the data
plt.figure(figsize=(14, 6))
plt.plot(data['Jahr'], data['Alter'].str.replace(',', '.'), marker='o', linestyle='-', color='black')
plt.xlabel('Jahr', fontsize=18, fontweight='bold', fontname='Arial')
plt.ylabel('Alter der Mutter in Jahren', fontsize=18, fontweight='bold', fontname='Arial')
plt.xticks(data['Jahr'], fontsize=16, fontname='Arial')  # Showing only whole years
plt.yticks(fontsize=16, fontname='Arial')
plt.grid(True)
plt.show()
