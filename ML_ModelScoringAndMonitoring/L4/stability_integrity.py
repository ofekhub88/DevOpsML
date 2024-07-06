import ast
import pandas as pd


with open('historicmeans.txt', 'r') as f:
    histmeanslist = ast.literal_eval(f.read())

newdata = pd.read_csv("samplefile2.csv")

newmeans=list(newdata.mean())
print("meandata",newmeans)
meancomparison=[(newmeans[i]-histmeanslist[i])/histmeanslist[i] for i in range(len(histmeanslist))]
print(meancomparison)

nas=list(newdata.isna().sum())
col_na_precent = []
for col in nas:
  col_na_precent.append(col/len(newdata.index))
print(col_na_precent)
