import pandas as pd

# Load the datasets with appropriate delimiter
marktanteile_absolut = pd.read_csv('Marktanteile_absolut.csv', delimiter=';')
absatzzahlen_a_neu = pd.read_csv('Absatzzahlen_A_neu.csv', delimiter=';')
absatzzahlen_a_gefiltert = pd.read_csv('Absatzzahlen_A_gefiltert.csv', delimiter=';')
eu_statistics_age_mom = pd.read_csv('eu_statistics_age_mom.csv', delimiter=';')

# Convert the 'Alter' column in eu_statistics_age_mom to numeric, replacing commas with dots
eu_statistics_age_mom['Alter'] = eu_statistics_age_mom['Alter'].str.replace(',', '.').astype(float)

# Extract only the relevant columns 'Summe Monat' and 'A' for Marktanteile Absolut
marktanteile_absolut_relevant = marktanteile_absolut[['Summe Monat', 'A']]

# Describe the datasets
marktanteile_absolut_relevant_desc = marktanteile_absolut_relevant.describe()
absatzzahlen_a_neu_desc = absatzzahlen_a_neu.describe()
absatzzahlen_a_gefiltert_desc = absatzzahlen_a_gefiltert.describe()
eu_statistics_age_mom_desc = eu_statistics_age_mom.describe()

# Display the descriptions
print("Marktanteile Absolut (Gesamtmarkt und Marktanteil A) Description:")
print(marktanteile_absolut_relevant_desc)

print("\nAbsatzzahlen A Neu Description:")
print(absatzzahlen_a_neu_desc)

print("\nAbsatzzahlen A Gefiltert Description:")
print(absatzzahlen_a_gefiltert_desc)

print("\nEU Statistics Age Mom Description:")
print(eu_statistics_age_mom_desc)
