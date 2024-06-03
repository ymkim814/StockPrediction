# -*- coding: utf-8 -*-
"""Clustering

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1XllH-2Zwsdu2aJjCwNCahahBvVrIGHXK
"""

import os
import pandas as pd
!pip install yahoo_fin --upgrade
import yahoo_fin
import numpy as np
from yahoo_fin.stock_info import get_data
!pip install yfinance
from sklearn.model_selection import train_test_split
import yfinance as yf
import matplotlib.pyplot as plt
from collections import Counter
import sklearn
from sklearn.metrics import silhouette_score
from sklearn.manifold import MDS

# from pydrive.auth import GoogleAuth
# from pydrive.drive import GoogleDrive
# from google.colab import auth
# from oauth2client.client import GoogleCredentials

# # Authenticate and create the PyDrive client
# auth.authenticate_user()
# gauth = GoogleAuth()
# gauth.credentials = GoogleCredentials.get_application_default()
# drive = GoogleDrive(gauth)

# # Comment out to use GPU instead of CPU
#  tf.config.set_visible_devices([], 'GPU')

np.random.seed(5)

s_p = pd.read_csv('constituents.csv')
full_ticker = s_p['Symbol'].tolist()

# 30 sec
ticker_info = pd.DataFrame(columns=['Ticker', 'Industry', 'Sector'])

for each_ticker in full_ticker:
    stock = yf.Ticker(each_ticker)
    industry = stock.info.get('industry', 'Unknown')
    #print(industry)
    sector = stock.info.get('sector', 'Unknown')
    ticker_info = pd.concat([ticker_info, pd.DataFrame({'Ticker': [each_ticker], 'Industry': [industry], 'Sector': [sector]})], ignore_index=True)
print(ticker_info)

ticker_counts = ticker_info.groupby('Sector')['Ticker'].count().sort_values(ascending=False)
ticker_counts

ticker_info.groupby('Sector')['Industry'].nunique()
unique_industry = ticker_info.groupby('Sector')['Industry'].nunique().sort_values(ascending=False)
unique_industry

#8 minutes ...
stock_tickers = full_ticker # Add more tickers as needed
stock_full = []
error_count = 0

for ticker in stock_tickers:
    try:
        stock_data = get_data(ticker, start_date="2017-01-01", end_date="2019-12-31", index_as_date=True, interval="1d")
        #stock_data['SMA_20'] = stock_data['close'].rolling(window=20).mean()
        #stock_data['SMA_50'] = stock_data['close'].rolling(window=50).mean()
        stock_data = stock_data.sort_index(ascending=False)
        stock_full.append(stock_data)
    except Exception as e:
        print(f"Error occurred for {ticker}: {e}")
        error_count +=1

print(error_count)

#   open        high         low       close    adjclose volume

# remove the inconsist data - that does not have length 753 (# of time stamp)
stock_full_res = [i for i in  stock_full if len(i) == 753]
print(len(stock_full_res))

stock_full_res_array = np.array(stock_full_res)

# Pull out tickers
tickers = stock_full_res_array[:, :, -1].flatten()

# Remove the ticker values along the third dimension
stock_full_res_without_tickers = np.delete(stock_full_res_array, -1, axis=2)

#stock_array = np.array([df.values for df in stock_full_res_without_tickers])
stock_array = stock_full_res_without_tickers
stock_array = stock_array.astype(float)

column_means = np.nanmean(stock_array, axis=(0, 1))

# Create a mask for NaN values in the array
nan_mask = np.isnan(stock_array)

# Replace NaN values with column means
for i in range(stock_array.shape[2]):  # Iterate over columns
    stock_array[:, :, i][nan_mask[:, :, i]] = column_means[i]

pip install tslearn

# Replace infinity or very large values with a finite maximum value
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

max_value = np.finfo(np.float64).max  # Maximum finite value for float64
stock_array[np.isinf(stock_array) | (np.abs(stock_array) > max_value)] = max_value
X_train, X_test = train_test_split(stock_array, test_size=0.2, random_state=42)

n_clusters = 12
#y_pred = model.fit_predict(X_train)
stock_array = TimeSeriesScalerMeanVariance().fit_transform(stock_array)

# Fit TimeSeriesKMeans with preprocessed data
model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", verbose = True,  max_iter_barycenter=20, max_iter=50, random_state=5)
y_pred = model.fit_predict(stock_array)

# Predict clusters for test data
#test_cluster_labels = model.predict(X_test)
# Retrieve the centroids of the clusters
centroids = model.cluster_centers_
centroids.shape

y_pred

# MDS

# Would it be better if I reduce

n_samples, seq_length, n_features = stock_array.shape
stock_array_reshaped = stock_array.reshape((n_samples, seq_length * n_features))

mds = MDS(n_components=2)
X_train_mds = mds.fit_transform(stock_array_reshaped)

plt.figure(figsize=(8, 6))

for i in range(n_clusters):
    plt.scatter(X_train_mds[y_pred == i, 0], X_train_mds[y_pred == i, 1], label=f'Cluster {i}')

#plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', s=200, c='red', label='Centroids')
plt.title('Time Series Clustering (MDS)')
plt.xlabel('MDS Component 1')
plt.ylabel('MDS Component 2')
plt.legend()
plt.show()

#company , timetrend label, x, y
#company, industry labell, x, y

# Import Counter from collections module

# Count the occurrences of each cluster label
cluster_counts = Counter(y_pred)

# Sort cluster counts by cluster ID
sorted_cluster_counts = sorted(cluster_counts.items())

# Begin the LaTeX table
latex_table = r"""\begin{table}[htbp]
    \centering
    \begin{tabular}{|c|c|}
        \hline
        \textbf{Cluster ID} & \textbf{Number of Stocks} \\
        \hline
"""

# Iterate over the sorted cluster counts and add rows to the LaTeX table
for cluster_id, count in sorted_cluster_counts:
    latex_table += f"        {cluster_id} & {count} \\\\\n"

# Complete the LaTeX table
latex_table += r"""        \hline
    \end{tabular}
    \caption{Number of stocks in each cluster}
    \label{tab:cluster_counts}
\end{table}
"""

print(latex_table)

cluster_info_df = pd.DataFrame(columns=["Ticker", "Cluster", "Sector", "Industry"])

# Group by cluster and print ticker information
for i, cluster_id in enumerate(y_pred):
    ticker = full_ticker[i]
    #print(f"Processing {ticker}...")
    stock = yf.Ticker(ticker)
    sector = stock.info.get('sector', 'N/A')
    industry = stock.info.get('industry', 'N/A')
    cluster_info_df = pd.concat([cluster_info_df, pd.DataFrame({"Ticker": [ticker],
                                                               "Cluster": [cluster_id],
                                                               "Sector": [sector],
                                                               "Industry": [industry]})],
                                ignore_index=True)

sector_cluster_mapping = {
    'Industrials': 0,
    'Healthcare': 1,
    'Technology': 2,
    'Utilities': 3,
    'Financial Services': 4,
    'Basic Materials': 5,
    'Consumer Cyclical': 6,
    'Real Estate': 7,
    'Communication Services': 8,
    'Consumer Defensive': 9,
    'Energy': 10,
    'N/A': 11
}

# Map the sectors to cluster numbers and create a new column 'Sector_Cluster'
#test_cluster_info_df['Sector_Cluster'] = test_cluster_info_df['Sector'].map(sector_cluster_mapping)
cluster_info_df['Sector_Cluster'] = cluster_info_df['Sector'].map(sector_cluster_mapping)

# Print the updated DataFrame
#print(test_cluster_info_df)

y_sector = cluster_info_df['Sector_Cluster']

plt.figure(figsize=(8, 6))

for i in range(n_clusters):
    plt.scatter(X_train_mds[np.array(y_sector) == i, 0], X_train_mds[np.array(y_sector) == i, 1], label=f'Cluster {i}')

#plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', s=200, c='red', label='Centroids')
plt.title('Sector Clustering (MDS)')
plt.xlabel('MDS Component 1')
plt.ylabel('MDS Component 2')
plt.legend()
plt.show()

y_section = np.array(y_sector)

sklearn.metrics.silhouette_score(y_pred.reshape(-1,1), y_section)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
.todevice()