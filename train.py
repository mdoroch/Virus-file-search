import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MultiLabelBinarizer
import joblib

df_train = pd.read_table('train.tsv')
df_train_target = df_train['is_virus']
df_test = pd.read_table('test.tsv')
df_val = pd.read_table('val.tsv')
df_val_target = df_val['is_virus']

df_train = df_train['libs']
df_val = df_val['libs']
df_test = df_test['libs']

tokens = []
for ind,st in enumerate(df_train):
  st_new = st.split(",")
  for word in st_new:
    tokens.append(word)
  df_train[ind]=st_new
df_train.columns = ['Id','Libs']

for ind,st in enumerate(df_test):
  st_new = st.split(",")
  df_test[ind]=st_new
df_test.columns = ['Id','Libs']

for ind,st in enumerate(df_val):
  st_new = st.split(",")
  df_val[ind]=st_new
df_val.columns = ['Id','Libs']

tokens = set(tokens)

new_pd = pd.DataFrame(columns=tokens)
new_pd.insert(loc=0, column='Id', value=np.arange(0,16290,1))
new_pd = new_pd.set_index('Id')
mlb = MultiLabelBinarizer(classes=new_pd.columns)


df_train = pd.DataFrame(
    mlb.fit_transform(df_train),
    index=df_train.index,
    columns=mlb.classes_
    )
df_val = pd.DataFrame(
    mlb.fit_transform(df_val),
    index=df_val.index,
    columns=mlb.classes_
    )
df_test = pd.DataFrame(
    mlb.fit_transform(df_test),
    index=df_test.index,
    columns=mlb.classes_
    )


selector = SelectKBest(chi2, k=900)
selector.fit(df_train, df_train_target)
cols = selector.get_support(indices=True)
df_train = df_train.iloc[:,cols]
df_val = df_val.iloc[:,cols]
df_test = df_test.iloc[:,cols]

df_train.to_pickle('df_train_prep.pkl')
df_val.to_pickle('df_val_prep.pkl')
df_test.to_pickle('df_test_prep.pkl')
df_train_target.to_pickle('df_train_target_prep.pkl')
df_val_target.to_pickle('df_val_target_prep.pkl')

clf = LogisticRegression(max_iter=1000)
clf.fit(df_train,df_train_target)

joblib.dump(clf, "model.pkl")
