import pandas as pd
import numpy as np

#specify paths to the different csv files
paths = ['Zensus_Bevoelkerung_100m-Gitter.csv','Bevoelkerung100M.csv','Haushalte100m.csv','Familie100m.csv','Geb100m.csv','Wohnungen100m.csv']

#read translation excel
translations = pd.read_excel('/home/jovyan/work/germany/Data_Format_Census.xlsx', sheet_name='Translations')
#tansform to dictionary with original as key and translation as value
translation_dict = translations.set_index('Original')['Translated'].to_dict()

for i in paths:
    print(i)
    #read in data as dataframes
    if i in ['Zensus_Bevoelkerung_100m-Gitter.csv','Bevoelkerung100M.csv']:
        data = pd.read_csv("/home/jovyan/work/germany/raw_data/" + i, 
                       delimiter=';', encoding="cp1252")
    else:
        data = pd.read_csv("/home/jovyan/work/germany/raw_data/" + i, 
                        delimiter=',', encoding="cp1252")
    
    if i == 'Zensus_Bevoelkerung_100m-Gitter.csv':
        #replace all -1 with NaN
        data = data.replace(-1, np.nan)
        
        #rename columns
        data = data.rename(columns=translation_dict)

    else:
        #pivot dataframe to turn rows into columns
        data = data.pivot(index='Gitter_ID_100m',columns=['Merkmal', 'Auspraegung_Text'], values='Anzahl')

        #rename columns
        data = data.rename(columns=translation_dict)
        data = data.rename_axis(columns=translation_dict)
        data.index.names =  ['ID']
    
    data.to_parquet('/home/jovyan/work/germany/processed_data/'+ i[:-4] + '.parquet')
