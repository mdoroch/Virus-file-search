import joblib
import pandas as pd

df_test = pd.read_pickle('df_test_prep.pkl')
clf = joblib.load("model.pkl")
y_pred = clf.predict(df_test)
with open('prediction.txt', 'a') as f:
    f.write('prediction' + "\n")
    for elem in y_pred:
        f.write(str(elem) + '\n')
    f.close



