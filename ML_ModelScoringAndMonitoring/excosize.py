import pandas as pd
from datetime import datetime
import os 

sourcelocation=['/L2/data1','/L2/data2','/L2/data3']
final_dataframe = pd.DataFrame(columns=['peratio','price'])
for dir in sourcelocation:
    filenames = os.listdir(os.getcwd()+dir)
    for file in filenames:
       currentdf = pd.read_csv(os.getcwd()+dir+"/"+file)
       final_dataframe=final_dataframe.append(currentdf).reset_index(drop=True)
    #final_dataframe=final_dataframe.append(currentdf).reset_index(drop=True)
print(final_dataframe.duplicated())
print("size before",final_dataframe.size)
final_dataframe.drop_duplicates( inplace=True)
print("size before",final_dataframe.size)
   
    
final_dataframe.to_csv("result.csv",index=False)
 

dateTimeObj=datetime.now()
thetimenow=str(dateTimeObj.year)+ '/'+str(dateTimeObj.month)+ '/'+str(dateTimeObj.day)



#allrecords=[filenames,len(data.index),thetimenow]


#MyFile=open(outputlocation,'w')
#for element in allrecords:
#     MyFile.write(str(element))