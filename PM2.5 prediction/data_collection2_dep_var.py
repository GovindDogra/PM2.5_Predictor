import pandas as pd

def avg_data(year):
    temp_i=0
    average=[]
    file='data/AQI/aqi{}.csv'.format(year)
    for rows in pd.read_csv(file,chunksize=24):
        avg=0.0
        add_var=0
        data=[]
        df=pd.DataFrame(data=rows)
        for index,row in df.iterrows():
            data.append(row['PM2.5'])
        for i in data:
            if type(i) is float or type(i) is int:
                add_var=add_var+i
            elif type(i) is str:
                if i!='NoData' and i!='PwrFail' and i!='---' and i!='InVld':
                    temp=float(i)
                    add_var=add_var+temp
        avg=add_var/24
        temp_i=temp_i+1
            
        average.append(avg)
    return average
        
if __name__=='__main__':
    for year in range(2013,2019):
        if year == 2013:
            lst2013=avg_data(year)
        elif year == 2014:
            lst2014=avg_data(year)
        elif year == 2015:
            lst2015=avg_data(year)
        elif year == 2016:
            lst2016=avg_data(year)
        elif year == 2017:
            lst2017=avg_data(year)
        elif year == 2018:
            lst2018=avg_data(year)
        
    