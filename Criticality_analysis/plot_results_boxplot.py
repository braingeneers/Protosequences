import seaborn as sns
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams.update({'font.family': 'Arial'})

today = date.today()
today = today.strftime("%Y_%m_%d")

# Load data
df = pd.read_csv('d2_nocriter.csv', index_col=0)
# Set up the matplotlib figure
fig, axes = plt.subplots(1, 4, figsize=(20, 10), sharey=True)

# Flatten axes for easy indexing
axes = axes.flatten()

# Define color palette for "Type" (Control vs Intact) and "OrganoidName"
type_palette = {'Intact': '#81d8d0', 'Control': '#ff7373'}
type_palette_d = {'Intact': '#68aea7', 'Control': '#cc5c5c'}
organoid_palette = sns.color_palette("Set2", len(df['OrganoidName'].unique()))

# Plot each OrganoidName vs Type with colors
organoid_names = df['OrganoidName'].unique()

for i, organoid_name in enumerate(organoid_names):
    sns.boxplot(x='Type', y='d2',
                data=df[df['OrganoidName'] == organoid_name],
                ax=axes[i], palette=type_palette)
    sns.stripplot(x='Type', y='d2',
                  data=df[df['OrganoidName'] == organoid_name],
                  ax=axes[i], palette=type_palette_d)
    axes[i].set_ylabel('$d_2$')
    axes[i].set_xlabel(df.loc[df.OrganoidName == organoid_name,
                              'OrganoidDescription'].tolist()[0])
    axes[i].grid()

# Adjust layout to avoid overlap
plt.tight_layout()
plt.show()
