import numpy as np
import seaborn as sbs
import matplotlib.pyplot as plt
import pandas as pd
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2

dataset = pd.read_csv("./student/student-por.csv", sep=';')
# print(dataset.head())
# print(dataset.dtypes)
# https://pbpython.com/categorical-encoding.html
obj_df = dataset.select_dtypes(include=['object']).copy()
cleanup_nums = {"school": {"GP": 1, "MS": 0},
                "sex": {"F": 1, "M": 0},
                "address": {"U": 1, "R": 0},
                "famsize": {"LE3": 1, "GT3": 0},
                "Pstatus": {"A": 1, "T": 0},
                "Mjob": {"teacher":1, "health":2, "services":3, "at_home":4, "other":5},
                "Fjob": {"teacher":1, "health":2, "services":3, "at_home":4, "other":5},
                "reason": {"home":1, "reputation":2, "course":3, "other":4},
                "guardian": {"mother":1, "father":2, "other":3},
                "schoolsup": {"yes": 1, "no": 0},
                "famsup": {"yes":1, "no":0},
                "paid": {"yes":1, "no":0},
                "activities": {"yes":1, "no":0},
                "nursery": {"yes":1, "no":0},
                "higher": {"yes":1, "no":0},
                "internet": {"yes":1, "no":0},
                "romantic": {"yes":1, "no":0}}
obj_df.replace(cleanup_nums, inplace=True)
# print(obj_df.head())
# age
age_col = dataset['age'].copy()
def age(x):
    if x <= 18:
        return 'young'
    else:
        return 'elder'
newAge = age_col.apply(age)
# absences
ab_col = dataset['absences'].copy()
# print(ab_col.mean()) about 3.7 
def ab(x):
    if x <= 3.7:
        return 'low'
    else:
        return 'high'
newab = ab_col.apply(ab)
# print(newab.head())
# 0-20; use 60% as pass; 20*0.6=12; np: not pass
# G1
g1_col = dataset['G1'].copy()
def g1(x):
    if x <= 12:
        return 'np'
    else:
        return 'pass'
newg1 = g1_col.apply(g1)
# print(newg1.head())
# G2
g2_col = dataset['G2'].copy()
def g2(x):
    if x <= 12:
        return 'np'
    else:
        return 'pass'
newg2 = g2_col.apply(g2)
# print(newg2.head())
# G3
g3_col = dataset['G3'].copy()
def g3(x):
    if x <= 12:
        return 'np'
    else:
        return 'pass'
newg3 = g3_col.apply(g3)
# print(newg3.head())

# numeric_1 = pd.concat([newAge, newab, newg1, newg2, newg3], axis=1)
numeric_1 = pd.concat([newAge, newab], axis=1)
# print(numeric_1.head())

nm_df = numeric_1.copy()
# cleanup_num1 = {"age": {"young": 1, "elder": 0},
#                 "absences": {"high": 1, "low": 0},
#                 "G1": {"pass": 1, "np": 0},
#                 "G2": {"pass": 1, "np": 0},
#                 "G3": {"pass": 1, "np": 0}}
cleanup_num1 = {"age": {"young": 1, "elder": 0},
                "absences": {"high": 1, "low": 0}}
nm_df.replace(cleanup_num1, inplace=True)
# print(nm_df.head())

int_df = dataset.select_dtypes(include=['int64']).copy()
# int_df = int_df.drop(columns=['age', 'absences','G1','G2','G3'])
int_df = int_df.drop(columns=['age', 'absences'])
# print(int_df.head())

finalXtable = pd.concat([int_df, obj_df, nm_df], axis=1)
# print(finalXtable.head())
# print(dataset.dtypes)
# print(type(finalXtable))

# draw heatmap
# hm = plt.figure(figsize=(15, 15))
# hm = sbs.heatmap(finalXtable.corr(), annot=True, square=True, annot_kws={"fontsize":6})
# plt.show()

finalXtable.to_csv(r'Xtable.txt', header=True, index=False, sep=' ', mode='w')
train_cols = finalXtable.drop(columns=['G3'])
y_col = finalXtable['G3']
# print(train_cols.head())
# print(y_col.head())

# arrayX = train_cols.values
# arrayY = y_col.values
# X_new = SelectKBest(chi2, k=5).fit_transform(arrayX, arrayY)
# print(X_new[:10])
