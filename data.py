import requests
import os
import zipfile
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler


# URL列表
urls = [
    "https://media.fdj.fr/static-draws/csv/euromillions/euromillions_202002.zip",
    "https://media.fdj.fr/static-draws/csv/euromillions/euromillions_201902.zip",
    "https://media.fdj.fr/static-draws/csv/euromillions/euromillions_201609.zip",
    "https://media.fdj.fr/static-draws/csv/euromillions/euromillions_201402.zip",
    "https://media.fdj.fr/static-draws/csv/euromillions/euromillions_201105.zip",
    "https://media.fdj.fr/static-draws/csv/euromillions/euromillions_200402.zip"
]

# 存储路径
save_path = '/Users/dantashashou/Downloads/euromillions'  # 请根据实际情况调整路径

# 检查目标目录是否存在，如果不存在，则创建
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 下载并解压文件
for url in urls:
    file_name = url.split('/')[-1]
    full_path = os.path.join(save_path, file_name)

    # 发起请求下载文件
    response = requests.get(url)
    if response.status_code == 200:
        # 写入文件到指定路径
        with open(full_path, 'wb') as file:
            file.write(response.content)
        print(f"文件已下载并保存到：{full_path}")
        
        # 解压文件
        with zipfile.ZipFile(full_path, 'r') as zip_ref:
            zip_ref.extractall(save_path)
        print(f"文件已解压到：{save_path}")
        
        # 删除原zip文件
        os.remove(full_path)
        print(f"原zip文件已删除：{full_path}")
    else:
        print(f"下载失败，状态码：{response.status_code}")


# Define the folder path where the CSV files are located
folder_path = '/Users/dantashashou/Downloads/euromillions'

# Use the glob module to find all CSV files in the folder
csv_files = glob.glob(f'{folder_path}/*.csv')

# Define the column names that you want to keep
columns_to_keep = ['annee_numero_de_tirage', 'boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5', 'etoile_1', 'etoile_2']

# Initialize an empty DataFrame to hold data from all files
df = pd.DataFrame()

# Iterate over each CSV file
for file in csv_files:
    # Read the CSV file, only keeping specified columns
    df_temp = pd.read_csv(file, encoding='ISO-8859-1', sep=';', usecols=columns_to_keep)
    # Append the data to the combined DataFrame
    df = pd.concat([df, df_temp], ignore_index=True)

# Sort and reset index
df.sort_values(by='annee_numero_de_tirage', inplace=True)
df.reset_index(drop=True, inplace=True)
df.iloc[:, 1:6] = np.sort(df.iloc[:, 1:6].values, axis=1)
df.iloc[:, 6:8] = np.sort(df.iloc[:, 6:8].values, axis=1)

# Basic statistics
stats = df.describe()
print(stats)

# Boxplot for detecting potential outliers
plt.figure(figsize=(14, 7))
sns.boxplot(data=df[['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5', 'etoile_1', 'etoile_2']])
plt.title('Boxplot of Lottery Numbers')
plt.show()

# Frequency distributions
for i in range(1, 6):  # Main balls
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df[f'boule_{i}'], nbinsx=50, marker_color='blue', name=f'boule_{i}'))
    fig.update_layout(title_text=f'Frequency of boule_{i}', xaxis_title_text='Number', yaxis_title_text='Frequency', bargap=0.2)
    fig.show()

for i in range(1, 3):  # Star balls
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df[f'etoile_{i}'], nbinsx=12, marker_color='red', name=f'etoile_{i}'))
    fig.update_layout(title_text=f'Frequency of etoile_{i}', xaxis_title_text='Number', yaxis_title_text='Frequency', bargap=0.2)
    fig.show()

# Correlation heatmap
corr_matrix = df[['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='viridis', fmt=".2f")
plt.title('Heatmap of Correlation Between Main Balls')
plt.show()

# Standardize the dataset
df = df.drop(['annee_numero_de_tirage'], axis=1)
scaler = StandardScaler().fit(df.values)
transformed_dataset = scaler.transform(df.values)
transformed_df = pd.DataFrame(data=transformed_dataset, columns=df.columns)

# Export transformed_df to a CSV file
transformed_df.to_csv('transformed_data.csv', index=False)
