import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from scipy.io import arff
from matplotlib.patches import Ellipse

dataset = arff.loadarff('./php50jXam.arff')
dataset = pd.DataFrame(dataset[0])

# Convert "Class" column to string values
dataset['Class'] = dataset['Class'].str.decode('utf-8')

dataset['Class'] = [x.encode('utf-8') for x in dataset['Class']]

def decode_class(x):
  return x.decode('utf-8')

dataset['Class'] = dataset['Class'].apply(decode_class)


dataset.sort_values('V1', inplace=True)

print(dataset)

v1 = dataset['V1']
v2 = dataset['V2']
classes = dataset['Class']

print(v1)
print(v2)
print(classes)


print(dataset)
datas_classe1 = dataset[dataset['Class'] == '1']
datas_classe2 = dataset[dataset['Class'] == '2']
print('----------Genuine Notes data----------')
print(datas_classe1)

print('---------Clean datas-----------')
datas1 = datas_classe1[['V1', 'V2', 'Class']]
datas2 = datas_classe2[['V1', 'V2', 'Class']]

print('---------------Means----------------')
print('--------Genuine Notes-------')
mean_datas1_v1 = np.mean(datas1['V1'])
mean_datas1_v2 = np.mean(datas1['V2'])
print(mean_datas1_v1)
print(mean_datas1_v2)
print('--------Fake Notes-------')
mean_datas2_v1 = np.mean(datas2['V1'])
mean_datas2_v2 = np.mean(datas2['V2'])
print(mean_datas2_v1)
print(mean_datas2_v2)

print('---------------Standard Deviation----------------')
print('------Genuine Notes------')
std_datas1_v1 = np.std(datas1['V1'])
std_datas1_v2 = np.std(datas1['V2'])
print(std_datas1_v1)
print(std_datas1_v2)
print('------Fake Notes-------')
std_datas2_v1 = np.std(datas2['V1'])
std_datas2_v2 = np.std(datas2['V2'])
print(std_datas2_v1)
print(std_datas2_v2)

print('----------False data----------')
print(datas_classe2)

plt.scatter(mean_datas1_v1, mean_datas1_v2, c='red', s=50, alpha=1, label='Mean Genuine Notes')
plt.scatter(mean_datas2_v1, mean_datas2_v2, c='fuchsia', s=50, alpha=1, label='Mean Genuine Notes')

plt.xlabel('Variance of Wavelet')
plt.ylabel('Skewness of Wavelet')




def run_kmeans(data, color, bg_elps):
  km_results = KMeans(n_clusters=1).fit(data)
  cluster_center = km_results.cluster_centers_[0]
  ellipse = Ellipse(xy=cluster_center, width=2 * np.std(data['V1']), height=2 * np.std(data['V2']),
                    edgecolor=color, facecolor=bg_elps, alpha=0.3)
  plt.gca().add_patch(ellipse)

for _ in range(5):  # Executar o K-means 5 vezes como exemplo

  run_kmeans(datas1, 'lightsteelblue', 'mediumblue')
  run_kmeans(datas2, 'bisque', 'orange')

  plt.scatter(datas1['V1'], datas1['V2'], c='blue', s=30, alpha=0.9, label='Genuine Notes')
  plt.scatter(datas2['V1'], datas2['V2'], c='orange', s=30, alpha=0.9, label='Fake Notes')

  plt.legend(loc='lower right')
  plt.show()


