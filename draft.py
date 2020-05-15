import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

Importation de données

train_df = pd.read_csv('C:/Users/TAHIRI/ESGF/TP1/Data/train.csv')
test_df = pd.read_csv('C:/Users/TAHIRI/ESGF/TP1/Data/test.csv')

Préparation des jeux de données et formation du modèle


imageIds = list(range(1,28001))
X = train_df.drop('label', axis=1)
Y = train_df['label']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30)

Classer les images inconnues


clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, Y_train)  
classified = clf.predict(test_df)

print(classified)
[2 0 9 ... 3 9 2]

Exporter le résultat


df = pd.DataFrame()
df['ImageId'] = imageIds
df['Label'] = classified
df.to_csv('C:/Users/TAHIRI/ESGF/TP1/Data/submission.csv', index=False)
