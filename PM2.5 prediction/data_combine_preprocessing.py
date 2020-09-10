import requests 
import sys
from bs4 import BeautifulSoup
import pandas as pd
import csv
import numpy as np
import os
from data_collection2_dep_var import avg_data

def meta_data(month,year):
    file=open('Data/Html_Data/{}/{}.html'.format(year,month))
    temp_data=[]
    final_data=[]
    soup=BeautifulSoup(file,'html.parser')
    for table in soup.findAll('table',{'class':'medias mensuales numspan'}):  #in this we will find the <table> tag hasving class 'medias mensuales numspan ' as it contain all the info. we want 
        for tbody in table:
            for tr in tbody:
                    a= tr.getText()
                    temp_data.append(a)
    row=len(temp_data)/15
    
    for times in range(round(row)):
        tempd_a=[]
        for i in range(15):
            tempd_a.append(temp_data[0])
            temp_data.pop(0)
        final_data.append(tempd_a)
    
    final_data.pop(0)
    final_data.pop(len(final_data)-1)
    
    #changing my final_data list to a dataset
    final_data=pd.DataFrame(final_data,columns=['DAY','T','TM','Tm','SLP','H','PP','VV','V','VM','VG','RA','SN','TS','FG'])
    #dropping(deleting) empty and unwanted columns
    final_data=final_data.drop(['DAY','SLP','PP','VG','SN','TS','FG'],axis=1)
   
    #for coverting our "RA" in 0 and 1 form i.e 1 for rain and 0 for no rain 
    dd=final_data['RA']
    for i in range(len(dd)):
        if dd[i]=='o':
            dd[i]=1
        else:
            dd[i]=0
   
    final_data=final_data.values.tolist()
    return final_data 

def data_combine(year, cs):
    for a in pd.read_csv('Data/Real-Data/real_' + str(year) + '.csv', chunksize=cs):
        df = pd.DataFrame(data=a)
        mylist = df.values.tolist()
    return mylist
    
if __name__=='__main__':
    if not os.path.exists("Data/Real-Data"):
        os.makedirs("Data/Real-Data")
    for year in range(2013, 2019):
        cmplt_data = []
        with open('Data/Real-Data/real_' + str(year) + '.csv', 'w') as csvfile:
            wr = csv.writer(csvfile, dialect='excel')
            wr.writerow(['T', 'TM', 'Tm', 'H', 'VV', 'V', 'VM','RA', 'PM 2.5'])
        for month in range(1, 13):
            temp = meta_data(month,year)
            cmplt_data = cmplt_data + temp
          
        #getting values of dependent variable
        if year == 2013:
            pm=avg_data(year)
        elif year == 2014:
            pm=avg_data(year)
        elif year == 2015:
            pm=avg_data(year)
        elif year == 2016:
            pm=avg_data(year)
        elif year == 2017:
            pm=avg_data(year)
        elif year == 2018:
            pm=avg_data(year)
        

        for i in range(len(cmplt_data)-1):
            #cmplt_data[i].insert(0, i + 1)
            cmplt_data[i].insert(8, pm[i])
        #deleting the empty rows
        with open('Data/Real-Data/real_' + str(year) + '.csv', 'a') as csvfile:
            wr = csv.writer(csvfile, dialect='excel')
            for row in cmplt_data:
                flag = 0
                for elem in row:
                    if elem == "" or elem == "-":
                        flag = 1
                if flag != 1:
                    wr.writerow(row)
                    
    data_2013 = data_combine(2013, 600)
    data_2014 = data_combine(2014, 600)
    data_2015 = data_combine(2015, 600)
    data_2016 = data_combine(2016, 600)
    data_2017 = data_combine(2017, 600)
    data_2018 = data_combine(2018, 600)
    
     
    total=data_2013+data_2014+data_2015+data_2016+data_2017+data_2018
    
    with open('Data/Real-Data/Real_Combine.csv', 'w') as csvfile:
        wr = csv.writer(csvfile, dialect='excel')
        wr.writerow(
            ['T', 'TM', 'Tm', 'H', 'VV', 'V', 'VM','RA', 'PM 2.5'])
        wr.writerows(total)
        
        
df=pd.read_csv('Data/Real-Data/Real_Combine.csv')
            