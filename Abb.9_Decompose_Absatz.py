#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 12:44:24 2024

@author: carolinmagin
"""

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.dates as mdates

# Load the data
url = 'Absatzzahlen_A_neu.csv'
df = pd.read_csv(url, sep=';', parse_dates=['Datum'], dayfirst=True)
df.set_index('Datum', inplace=True)

# Set Arial font
plt.rcParams['font.family'] = 'Arial'

# Function to plot decomposition with customization
def plot_decomposition(result, color='red', scale=1):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12*scale, 8*scale), sharex=True)
    
    components = ['Observed', 'Trend', 'Seasonal', 'Residual']
    axes = [ax1, ax2, ax3, ax4]
    data = [result.observed, result.trend, result.seasonal, result.resid]
    
    for ax, component, d in zip(axes, components, data):
        d.plot(ax=ax, color=color, linewidth=2)  # Set line width to 2
        ax.text(-0.15, 0.5, component, transform=ax.transAxes, 
                fontsize=14*scale, va='center', ha='right', rotation=0, fontweight='bold')
        
        # Remove x-axis labels and ticks for all but the last subplot
        if ax != ax4:
            ax.set_xticklabels([])
            ax.tick_params(axis='x', which='both', bottom=False)
        
        # Remove all spines except the left and bottom ones
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add horizontal line to separate plots
        ax.axhline(y=ax.get_ylim()[0], color='black', linewidth=2)
        
    # Format x-axis to show only years on the bottom subplot
    ax4.xaxis.set_major_locator(mdates.YearLocator())
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax4.tick_params(axis='x', labelsize=12)
    
    plt.tight_layout()
    plt.show()

# Decompose the time series
result = seasonal_decompose(df['Summe von Verkaufte Einheiten'], model='additive', period=12)

# Plot the decomposition with customization
plot_decomposition(result, color='red', scale=1.2)  # You can adjust the scale factor as needed