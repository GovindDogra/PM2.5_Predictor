import os
import requests
import sys
import time

def getdata_html():
    for year in range(2013,2019):
        for month in range(1,13):
            if month<10:
                url='https://en.tutiempo.net/climate/0{}-{}/ws-420710.html'.format(month,year)
            else:
                url='https://en.tutiempo.net/climate/{}-{}/ws-420710.html'.format(month,year)
                
            texts=requests.get(url)#will download the data from given url
            utf_text=texts.text.encode('utf=8')#encoding the data 
            
            #creating folders and creating html files of data
            if not os.path.exists('C:/Users/Govind Dogra/Desktop/aqi/data/{}'.format(year)):
                os.makedirs('C:/Users/Govind Dogra/Desktop/aqi/data/{}'.format(year))
            with open('C:/Users/Govind Dogra/Desktop/aqi/data/{}/{}.html'.format(year,month),'wb') as html_file:
                html_file.write(utf_text)
                
            sys.stdout.flush()
        
if __name__=='__main__':
    start_time=time.time()
    getdata_html()
    stop_time=time.time()
    print('Time taken:{}'.format(stop_time-start_time))
    
    