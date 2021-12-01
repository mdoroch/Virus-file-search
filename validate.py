import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import joblib

df_val = pd.read_pickle('df_val_prep.pkl')
df_val_target = pd.read_pickle('df_val_target_prep.pkl')

clf2 = joblib.load("model.pkl")
y_pred = clf2.predict(df_val)

conf_matr = confusion_matrix(df_val_target, y_pred)

with open('validation.txt', 'a') as f:
    f.write('True positive:' + ' ' + str(conf_matr[0][0]) + "\n")
    f.write('False positive:' + ' ' + str(conf_matr[0][1]) + "\n")
    f.write('False negative:' + ' ' + str(conf_matr[1][0]) + "\n")
    f.write('True negative:' + ' ' + str(conf_matr[1][1]) + "\n")
    f.write('Accuracy:' + ' ' + str(accuracy_score(df_val_target, y_pred)) + "\n")
    f.write('Precision:' + ' ' + str(precision_score(df_val_target, y_pred)) + "\n")
    f.write('Recall:' + ' ' + str(recall_score(df_val_target, y_pred)) + "\n")
    f.write('F1:' + ' ' + str(f1_score(df_val_target, y_pred)))

    f.close

