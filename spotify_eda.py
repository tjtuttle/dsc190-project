# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# useful imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# %%
'''
Reading in the dataset
'''
df = pd.read_csv("data.csv")


# %%
'''
Each row is a song on Spotify
'''
print(f"There are {df.shape[0]} songs in the data, and {df.shape[1]} features per song")
df.head()
df.tail()


# %%
'''
Datatype inspection
'''
print(f"The datatypes of each feature: \n{df.dtypes}")


# %%
'''
Null value inspection
No null values are present
'''
print(f"Null values per feature: \n{df.isnull().apply(sum)}")


# %%
'''
Sample statistics
The bounds for most audio parameters are 0 to 1, interestingly there is at least one song where Spotify estimated the tempo as zero
The median year between 1920 and 2020 is 1970, however the mean year for a song on Spotify is 1977, implying that newer songs are better represented on Spotify
'''
print(f"{df.describe()}")


# %%
corr_matrix = np.abs(pd.DataFrame(
    df, df.index, df.columns[df.dtypes != 'object']).corr())
fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(corr_matrix)
plt.title("Absolute correlation between numeric features")
plt.show()


# %%
print(f"Correlation of numeric features with population: \n{np.abs(corr_matrix['popularity']).sort_values(ascending=False)}")


# %%
print(f"Number of unique artists: {df['artists'].nunique()}")
print(f"Mean number of songs per artist: {df['artists'].value_counts().mean()}")


# %%
df['number_of_songs_by_artist'] = df['artists'].map(
    df['artists'].value_counts())
df['artist_popularity'] = df['artists'].map(
    df.groupby('artists')['popularity'].mean())
corr_matrix = np.abs(pd.DataFrame(
    df, df.index, df.columns[df.dtypes != 'object']).corr())
print(
    f"Correlation of numeric features with population: \n{np.abs(corr_matrix['popularity']).sort_values(ascending=False)}")
'''
Artist popularity has a high correlation with the popularity of the song, as expected
'''


# %%
top_lists = [df.groupby('artists')['popularity'].sum().sort_values(ascending=False).head(20), df.groupby('artists')['popularity'].mean().sort_values(ascending=False).head(20)]
for i, top_combos in enumerate(top_lists):
    fig, axis = plt.subplots(figsize = (12, 10))
    axis = sns.barplot(x=top_combos.values, y=top_combos.index, orient="h", ax=axis)
    xlab = 'Total Popularity' if i == 0 else 'Average Popularity'
    title = 'Top Artist Combinations by Total Popularity' if i == 0 else 'Top Artist Combinations by Average Popularity'
    axis.set_xlabel(xlab)
    axis.set_ylabel('Artist Combination')
    axis.set_title(title)
    plt.show()


# %%
numeric_cols = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'valence',
                'liveness', 'speechiness', 'duration_ms', 'tempo', 'loudness', 'number_of_songs_by_artist']
for feat in numeric_cols:
        fig, axis = plt.subplots()
        feat_group =  df.groupby(feat)['popularity'].mean().to_frame().reset_index()
        axis = sns.scatterplot(feat_group[feat], feat_group['popularity'], ax=axis)
        axis.set_title(f'Mean Popularity vs {feat.capitalize()}')
        axis.set_ylabel('Mean Popularity', fontsize=10)
        axis.set_xlabel(feat.capitalize(), fontsize=10)
        plt.tight_layout()
        plt.show()


# %%
'''
Mean popularity of songs over time
'''
fig, axis = plt.subplots(figsize=(15, 4))
axis = df.groupby('year')['popularity'].mean().plot()
axis.set_title('Mean Popularity vs Year')
axis.set_ylabel('Mean Popularity')
axis.set_xlabel('Year')
axis.set_xticks(range(1920, 2021, 5))
plt.show()


# %%



