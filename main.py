import pandas as pd
import seaborn as sns

#read excel file into pandas dataframe
df = pd.read_excel('C:/Users/dtgiv/Projects/Data/2023_rookie_offensive_stats.xls')

df.head(20)

#preview last 10 rows
df.tail(10)

#get dimensions of dataframe
df.shape

#get a count of each unique player age
df['Age'].value_counts()

#get a count of the number of rookies on each team
df['Tm'].value_counts()

#get a count of players per number of plate appearances
df['PA'].value_counts()

#get average number of plate appearances
df['PA'].mean()

#view missing data per column
df.isnull().sum()

#limit df to only relevant offensive statistics categories
df = df[["Name", "ASG", "G", "PA", "R", "H", "2B", "3B", "HR", "RBI", "SB", "CS", "BB", "SO", "BA", "OBP",
          "SLG", "OPS", "Age", "Debut", "Tm"]]

df

#filter dataframe to only those players who debuted in 2023
df = df[df["Debut"] >= '2023-01-01']

df

#filter df to only players with at least 10 plate appearances
df = df[df["PA"] >= 10]

df

#delete totals row
df = df.drop(df.index[-1])

df

#view missing data per column
df.isnull().sum()

#start of clustering code
#encode teams
tmdf = df['Tm']

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df['Tm'] = le.fit_transform(df['Tm'])

tmdf = le.transform(tmdf)

tmdf

#visualize data

sns.scatterplot(data = df, x = 'Tm', y = 'H')

#create dataset for clustering analysis

clusterdf = df.iloc[:, [5, 20]]

clusterdf

#use elbow method to find optimal number of clusters

from sklearn.cluster import KMeans

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(clusterdf)
    wcss.append(kmeans.inertia_)

import matplotlib.pyplot as plt

plt.figure(figsize = (10, 5))
sns.lineplot(range(1, 11), wcss, marker = 'o', color = 'red')
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

#fit k-means to the data using 3 clusters and 300 iterations

kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

kmeans.fit(clusterdf)

#visualize clustered data
sns.scatterplot(data = clusterdf, x = 'Tm', y = 'H', hue = kmeans.labels_)

#attach cluster labels to clusterdf
clusterdf['cluster'] = kmeans.labels_

print(clusterdf.to_string())

clusterdf['cluster'].value_counts()

#view Cincinnati Reds players
clusterdf[clusterdf['Tm'] == 6]

#based on these results, 2 Reds players were clustered into the most elite group (0), 2 Reds players were clustered
#into the second most elite group (2), and 1 Reds player was clustered into the least elite group (1)

#let's see which other teams had players that were clustered into the most elite group, and how many
clusterdf[clusterdf['cluster'] == 0]

#no other team had more than one player clustered into the most elite group

#let's see which other teams had players that were clustered into the second most elite group, and how many
clusterdf[clusterdf['cluster'] == 2]

#it appears Tm 24 had 4 players clustered into this group, Tm 5 had 2 players clustered into this group, Tm 21
#had 3 players clustered into this group, and Tm 15 had 3 players clustered into this group

#apply k-means algorithm again using the same data with 2 clusters, same number of iterations
clusterdf2 = clusterdf.drop(['cluster'], axis = 1)

clusterdf2

#fit k-means to the data using 2 clusters and 300 iterations

kmeans = KMeans(n_clusters = 2, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

kmeans.fit(clusterdf2)

#visualize clustered data
sns.scatterplot(data = clusterdf2, x = 'Tm', y = 'H', hue = kmeans.labels_)

#attach cluster labels to clusterdf2
clusterdf2['cluster'] = kmeans.labels_

print(clusterdf2.to_string())

#view Cincinnati Reds players
clusterdf2[clusterdf2['Tm'] == 6]

#based on the results, the Reds had 3 players clustered into the most elite group (1), and 2 players
#clustered into the lower group (0)

#let's see which other teams had players that were clustered into the most elite group, and how many
clusterdf2[clusterdf2['cluster'] == 1]

#Tm 24 = 4 players
#Tm 15 = 4 players
#Tm 19 = 2 players

#apply k-means algorithm again using the same data with 4 clusters, same number of iterations
clusterdf4 = clusterdf.drop(['cluster'], axis = 1)

clusterdf4

#fit k-means to the data using 4 clusters and 300 iterations

kmeans = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

kmeans.fit(clusterdf4)

#visualize clustered data
sns.scatterplot(data = clusterdf4, x = 'Tm', y = 'H', hue = kmeans.labels_)

#looking at the scatterplot, I don't think 4 clusters makes sense for these data because the lowest numbers of hits
#are split into two clusters (0 & 2). However I think it is worth noting that the Reds (Tm 6) are the only team to have 2
#players clustered into the most elite group (1)

