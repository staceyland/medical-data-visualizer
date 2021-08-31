import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv("medical_examination.csv")

# Add 'overweight' column
df['overweight'] = (df['weight']/((df['height']/100) ** 2) > 25).astype(int)


# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
medical_dict = { 1: 0, 2: 1, 3: 1}
df['cholesterol'] = df['cholesterol'].map(medical_dict)
df['gluc'] = df['gluc'].map(medical_dict)

#df['cholesterol'] = df['cholesterol'].apply(lamba x: 0 if x == 1 else 1)
#df['gluc'] = df['gluc'].apply(lamba x: 0 if x == 1 else 1)

# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_cat = pd.melt(df, value_vars=['cholesterol', 'gluc', 'smoke','alco','active', 'overweight'], id_vars=['cardio'])


    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    df_cat["total"] = 1
    df_cat = df_cat.groupby(["cardio", "variable", "value"], as_index =False).count()

    # Draw the catplot with 'sns.catplot()'
    fig = sns.catplot(x='variable', y='total', data=df_cat, hue='value', col='cardio', kind='bar').fig


    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) & 
     (df['height'] >= df['height'].quantile(0.025)) &
     (df['height'] <= df['height'].quantile(0.975)) &
     (df['weight'] >= df['weight'].quantile(0.025)) & 
     (df['weight'] <= df['weight'].quantile(0.975))]


    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True




    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # Draw the heatmap with 'sns.heatmap()'

    sns.heatmap(corr, annot=True, fmt='.1f', mask=mask, vmin=.16, vmax=.32, center=0, square=True, linewidths=.5, cbar_kws={'shrink':.45, 'format':'%.2f'})

    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
