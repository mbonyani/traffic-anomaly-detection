
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder,LabelEncoder
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import time
from sklearn import metrics
import pickle
from beautifultable import BeautifulTable
from functions import pie_chart
from termcolor import colored



# kamel
def genral_preprocess(encoder_numeric,encoder_categoric,scaler_name,dataset_path,name_target,dataset_name):
 

  df= pd.read_csv(dataset_path)


  #******************************************* 
  num_row , num_column = df.shape



  #calculate number of classes
  classes = df[name_target].unique()
  num_class = len(classes)

  print(colored('*******Orginal label/Target info*******', 'green', attrs=['bold']))

  print(df[name_target].value_counts())

  print(colored('***************************************', 'green', attrs=['bold']))

  #determine which class is normal (is not anomaly)
  label = np.array(df[name_target])
  a,b = np.unique(label , return_counts=True)
  # print("a is:",a)
  # print("b is:",b)

  for i in range(len(b)):
    if b[i]== b.max():
      normal = a[i]
      #print('normal:', normal)
    if b[i] == b.min():
      unnormal = a[i]
      #print('unnorm:' ,unnormal)

  # show anomaly classes
  anomaly_class = []
  for f in range(len(a)): 
    if a[f] != normal:
      anomaly_class.append(a[f])
      #print(anomaly_class)
  
  # convert dataset classes to 2 classe: normal and unnormal
  label = np.where(label != normal, unnormal ,label)
  df[name_target]=label

  #showing columns's type: numerical or categorical
  numeric =0
  categoric = 0
  for i in range(df.shape[1]):
    df_col = df.iloc[:,i]
    if df_col.dtype == int and df.columns[i] != name_target:
      numeric +=1
    elif df_col.dtype == float and df.columns[i] != name_target:
      numeric += 1
    elif df.columns[i] != name_target:
      categoric += 1

  #replace labels with 0 and 1
  label = np.where(label == normal, 0 ,1)
  df[name_target]=label  


  delcolumn=[]
  addcolumn=[]

  #choose which type of encoders should apply on columns of data base on column types
  for i in range(df.shape[1]):
    df_col = df.iloc[:,i]
    if df_col.dtype == int:     
      if encoder_numeric == 'OrdinalEncoder': #,encoder_categoric
        encode = OrdinalEncoder()
        col_encoded1 = encode.fit_transform(df_col.values.reshape(-1,1))
        new_column = df.columns[i]
        df= df.assign(**{new_column:col_encoded1})

      elif encoder_numeric == 'LabelEncoder': #,encoder_categoric
        encode = LabelEncoder()
        col_encoded1 = encode.fit_transform(df_col.values.reshape(-1,1))
        new_column = df.columns[i]
        df= df.assign(**{new_column:col_encoded1})

      elif (encoder_numeric == 'OneHotEncoder') and df.columns[i] != name_target: 
        encode = OneHotEncoder(sparse=False)
        col_encoded1 = encode.fit_transform(df_col.values.reshape(-1,1))
        new_column = df.columns[i]
        df = df.drop(columns= new_column)
        for j in range(col_encoded1.shape[1]):
          name = new_column+str(j)
          df= df.assign(**{name:col_encoded1[:,j]})

      elif encoder_numeric == None: 
        pass

    elif df_col.dtype == float:       
      if encoder_numeric == 'OrdinalEncoder': #,encoder_categoric
        encode = OrdinalEncoder()
        col_encoded1 = encode.fit_transform(df_col.values.reshape(-1,1))
        new_column = df.columns[i]
        df= df.assign(**{new_column:col_encoded1})

      elif encoder_numeric == 'LabelEncoder': #,encoder_categoric
        encode = LabelEncoder()
        col_encoded1 = encode.fit_transform(df_col.values.reshape(-1,1))
        new_column = df.columns[i]
        df= df.assign(**{new_column:col_encoded1})

      elif (encoder_numeric == 'OneHotEncoder') and df.columns[i] != name_target: 
        encode = OneHotEncoder(sparse=False)
        col_encoded1 = encode.fit_transform(df_col.values.reshape(-1,1))
        new_column = df.columns[i]
        df = df.drop(columns= new_column)
        for j in range(col_encoded1.shape[1]):
          name = new_column+str(j)
          df= df.assign(**{name:col_encoded1[:,j]})

      elif encoder_numeric == None: 
        pass

    else:     
      if encoder_categoric == 'OrdinalEncoder': #,encoder_categoric
        encode = OrdinalEncoder()
        col_encoded1 = encode.fit_transform(df_col.values.reshape(-1,1))
        new_column = df.columns[i]
        df= df.assign(**{new_column:col_encoded1})

      elif encoder_categoric == 'LabelEncoder': #,encoder_categoric
        encode = LabelEncoder()
        col_encoded1 = encode.fit_transform(df_col.values.reshape(-1,1))
        new_column = df.columns[i]
        df= df.assign(**{new_column:col_encoded1})

      elif (encoder_categoric == 'OneHotEncoder') and df.columns[i] != name_target: 
        #print("HHHHHH")
        encode = OneHotEncoder(sparse=False)
        col_encoded1 = encode.fit_transform(df_col.values.reshape(-1,1))
        new_column = df.columns[i]
        delcolumn.append(new_column)
        addcolumn.append(col_encoded1)
        # df = df.drop(columns= new_column)
        # for j in range(col_encoded1.shape[1]):
        #   name = new_column+str(j)
        #   df= df.assign(**{name:col_encoded1[:,j]})
      
      elif encoder_categoric == None: 
        pass 

  df = df.drop(columns= delcolumn)

  for i in range(len(addcolumn)):
     for j in range(addcolumn[i].shape[1]):
          name = delcolumn[i]+str(j)
          df= df.assign(**{name:addcolumn[i][:,j]})

  # null_check: if more than half of a column was null, then that columns will be droped
  # otherwise if number of null was less than half of column, then nulls will replace with mean of that column
  test = []
  for i in range(df.shape[1]):
    if df.iloc[:,i].isnull().sum() > df.shape[0]//2:
      test.append(i)

    elif  df.iloc[:,i].isnull().sum() < df.shape[0]//2 and df.iloc[:,i].isnull().sum() != 0:

      m = df.iloc[:,i].mean()
      df.iloc[:,i] = df.iloc[:,i].replace(to_replace = np.nan, value = m)
  df = df.drop(columns=df.columns[test])


  # choose Standardization type
  if scaler_name == 'StandardScaler':
    for i in range(df.shape[1]):
      if df.columns[i] != name_target:
        df_col = df.iloc[:,i]
        std_scaler = StandardScaler()
        std_scaled_col1 = std_scaler.fit_transform(df_col.values.reshape(-1,1))
        new_column = df.columns[i]
        df= df.assign(**{new_column:std_scaled_col1})  

  elif scaler_name == 'MinMaxScaler':
    for i in range(df.shape[1]):
      if df.columns[i] != name_target:
        df_col = df.iloc[:,i]
        MinMax_scaler = MinMaxScaler()
        MinMax_scaled_col1 = MinMax_scaler.fit_transform(df_col.values.reshape(-1,1))
        new_column = df.columns[i]
        df= df.assign(**{new_column:MinMax_scaled_col1})   

  elif scaler_name == 'RobustScaler':
    for i in range(df.shape[1]):
      if df.columns[i] != name_target:
        df_col = df.iloc[:,i]
        Robust_scaler = RobustScaler()
        Robust_scaled_col1 = Robust_scaler.fit_transform(df_col.values.reshape(-1,1))
        new_column = df.columns[i]
        df= df.assign(**{new_column:Robust_scaled_col1})
  
  elif scaler_name == None:
    pass

    



  #calculate Anomaly_rate 
  b = df[name_target].value_counts()
  #print('b:',b)
  Anomaly_rate= b[1] / (b[0]+b[1])
  print(colored('******* contamination/Anomaly rate *******', 'green', attrs=['bold']))
  print(Anomaly_rate)
  contamination= float("{:.4f}".format(Anomaly_rate))
  print(colored('***************************************', 'green', attrs=['bold']))

  print(colored('****** binary label/Target info ******', 'green', attrs=['bold']))

  print(df[name_target].value_counts())

  print(colored('***************************************', 'green', attrs=['bold']))


  #*****************************pie

  unique_elements,counts_elements = np.unique(label , return_counts=True)
  pie_chart(dataset_name,counts_elements,unique_elements)

  #******************************

  #or
  # anomaly_rate = 1.0 - len(df.loc[df[name_target]=="jjjj"])/ len(df)
  # print("anomay rate 2:" , anomaly_rate)

  #rename labels column
  df = df.rename(columns = {'n' : 'binary_target'})  

  df.to_csv(f'/content/{dataset_name}.csv', index = False)  

  #********************************************table
  from beautifultable import BeautifulTable
  table = BeautifulTable(maxwidth=200)
  table.set_style(BeautifulTable.STYLE_RST)
  table.rows.append([num_row , num_column-1, '--', "{:.4f}".format(Anomaly_rate),
                     "{:.2f}".format(Anomaly_rate*100)+'%',numeric , categoric])
  table.rows.header = [dataset_name]
  table.columns.header = ['No. of Instances','No. of Features' , 'Anomaly class', 'Anomaly Rate(contamination)',
                          'Anomaly Percentage', 'No.of numerical col', 'No.of categorical']
  print(table)          



  return df,contamination