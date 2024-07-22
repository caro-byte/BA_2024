import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV files
file_path_age = 'eu_statistics_age_mom.csv'
file_path_sales = 'Absatzzahlen_A_neu.csv'

data_age = pd.read_csv(file_path_age, delimiter=';')
data_sales = pd.read_csv(file_path_sales, delimiter=';')

# Converting the 'Alter' column to numeric, replacing the comma with a dot
data_age['Alter'] = data_age['Alter'].str.replace(',', '.').astype(float)

# Converting the 'Datum' column to datetime format in sales data
data_sales['Datum'] = pd.to_datetime(data_sales['Datum'], format='%d.%m.%y')

# Creating a new dataframe with monthly data
monthly_data = data_sales.copy()
monthly_data.set_index('Datum', inplace=True)
monthly_data = monthly_data.resample('M').sum()

# Expanding the age data to monthly frequency
years = data_age['Jahr'].astype(int)
expanded_age_data = data_age.loc[data_age.index.repeat(12), 'Alter'].values[:len(pd.date_range(start=f'{years.min()}-01', end=f'{years.max()}-12', freq='M'))]

# Creating a new dataframe with monthly data
monthly_age_data = pd.DataFrame({
    'Datum': pd.date_range(start=f'{years.min()}-01', end=f'{years.max()}-12', freq='M'),
    'Alter': expanded_age_data
})

# Merge the datasets on the 'Datum' column
merged_data = pd.merge(monthly_data, monthly_age_data, on='Datum')

# Creating the scatter plot with the specified adjustments
plt.figure(figsize=(10.5, 5))
plt.scatter(merged_data['Summe von Verkaufte Einheiten'], merged_data['Alter'], color='black', marker='o', label='Datenpunkte')
plt.xlabel('Abgesetzte Units von A', fontsize=14, fontweight='bold', fontname='Arial')
plt.ylabel('Alter der Mutter', fontsize=14, fontweight='bold', fontname='Arial')
plt.xlim(merged_data['Summe von Verkaufte Einheiten'].min(), merged_data['Summe von Verkaufte Einheiten'].max())
plt.ylim(merged_data['Alter'].min() - 0.1, 30.2)
plt.yticks(np.arange(merged_data['Alter'].min() - 0.1, 30.2 + 0.1, 0.1))

# Adding a dotted grey trend line
z = np.polyfit(merged_data['Summe von Verkaufte Einheiten'], merged_data['Alter'], 1)
p = np.poly1d(z)
x_values = merged_data['Summe von Verkaufte Einheiten']
plt.plot(x_values, p(x_values), linestyle=':', color='grey')

# Adding text label for the trend line between 25,000 and 27,500 units
trend_x_position = 26000
trend_y_position = p(trend_x_position)
plt.text(trend_x_position, trend_y_position, 'Trend', fontsize=12, color='grey', ha='left')

plt.grid(True, axis='y')
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x)).replace(",", ".")))
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, loc: "{:.1f}".format(y).replace(".", ",")))

# Placing the legend below the plot (removing the legend as it is no longer necessary)
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

plt.show()

# Calculating the correlation coefficient between the two variables
correlation_coefficient = merged_data['Summe von Verkaufte Einheiten'].corr(merged_data['Alter'])

correlation_coefficient
