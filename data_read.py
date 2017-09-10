import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
def data_processing(a,date,enddate):
    a_itr = 0
    b = np.zeros(a.size, dtype='int32')
    if date=='2015-01-01':
        a_itr = 0
    else:
        for i in range(a.size):
            if np.datetime64(a._values[i]) == np.datetime64(date):
                a_itr = i
                break

    b_itr = 0
    data_thisyear = a[a_itr:]
    if date=='2015-01-01':
        srtdt = np.datetime64('2015-01-01')
    else:
        srtdt = np.datetime64(a[a_itr])
    dt_itr = 0
    while 1:
        count = 0
        while 1:

            if srtdt == np.datetime64(data_thisyear[a_itr + dt_itr]):
                count += 1
                dt_itr += 1
            else:
                srtdt += 1
                break
        if srtdt == (np.datetime64(enddate)+1):

            break
        b[b_itr] = count
        b_itr += 1
        sum_b=np.zeros(b_itr,dtype='int32')
        sum_b[0]=b[0]
        for i in range(1,b_itr):
            sum_b[i] +=sum_b[i-1]+b[i]
    return a_itr,b_itr,b,sum_b
def count_freq(b,b_itr):
    freq_count = np.zeros(9)
    for i in range(b_itr):
        freq_count[b[i]] += 1
    return freq_count
def poisson_test(x,freq_count,b_itr):

    k = sum(freq_count * x) / b_itr
    rv = st.poisson(k)
    predict = sum(freq_count) * rv.pmf(x)
    # dof =6
    X = np.empty(9)
    for i in range(9):
        X[i] = np.square(freq_count[i] - predict[i]) / predict[i]
    X2 = sum(X)
    return X2,predict,k
def plot_data(i,year,x,predict,freq_count):
    plt.figure(i)
    plt.plot(x, predict, ls='dashed', c='r', label='Expected data')
    plt.bar(x, freq_count, label='Observed data')
    plt.xlabel('Number of shoots happened in a day')
    plt.ylabel('Frequency')
    plt.title('Police shooting data '+year)
    plt.legend(loc='best')
    plt.savefig('data_'+year+'.jpg')
    plt.close()
def find_CI(b_itr,sum):
    #sum is the predicted value.
    #m=n
    L=np.zeros(b_itr)
    U=np.zeros(b_itr)
    for i in range(b_itr):
        #lmd=sum[i]/b_itr
        var=np.sqrt(sum[i]*2)
        L[i]=sum[i]-1.96*var
        U[i]=sum[i]+1.96*var
    return L,U
def CDF(k,itr):
    sum=np.zeros(itr)
    for i in range(itr):
        sum[i]=i*k
    return sum
def plot_cdf_predict(sum,L,U,year,itr,color):
    plt.plot(range(itr),sum,c=color,label='Predicted data '+year)
    plt.plot(range(itr),L,ls='dashed',c=color,label='95%CI')
    plt.plot(range(itr),U,ls='dashed',c=color)
    plt.xlabel('Date increasing')
    plt.ylabel(('Total shoot'))
    plt.legend(loc='best')
    plt.savefig('Sum of shooting '+year+'.jpg')
    #if year!='2015':
    #    plt.close()
    if year=='2017':

        plt.close()
    return None
#----------------------------------------------------------
if __name__=='__main__':
    df=pd.read_csv('data.csv')
    a=df.iloc[:,2]
    '''
    a_itr=0
    for i in range(a.size):
        if np.datetime64(a._values[i])==np.datetime64('2017-01-01'):
            a_itr = i
            break

    b_itr=0
    data_thisyear=a[a_itr:]
    srtdt=np.datetime64(a[a_itr])
    dt_itr=0
    while 1:
        count=0
        while 1:

            if srtdt==np.datetime64(data_thisyear[a_itr+dt_itr]):
                count += 1
                dt_itr += 1
            else:
                srtdt+=1
                break
        if srtdt==np.datetime64('2017-07-28'):
            break
        b[b_itr]=count
        b_itr+=1
        '''
    a_itr,b_itr,b,sum_b=data_processing(a,'2017-01-01','2017-07-27')# data 2017
    a_itr_2015,b_itr_2015,b_2015,sum_b_2015=data_processing(a,'2015-01-01','2015-12-31')
    a_itr_2016,b_itr_2016,b_2016,sum_b_2016=data_processing(a,'2016-01-01','2016-12-31')
    '''
    freq_count=np.zeros(8)
    for i in range(b_itr):
         freq_count[b[i]-1]+=1
         '''
    freq_count=count_freq(b,b_itr)
    freq_count_2015=count_freq(b_2015,b_itr_2015)
    freq_count_2016=count_freq(b_2016,b_itr_2016)
    '''x=range(1,10)
    k=sum(freq_count*x)/b_itr

    rv=st.poisson(k)
    predict=sum(freq_count)*rv.pmf(x)
    # dof =6
    X=np.empty(9)
    for i in range(9):
        X[i]=np.square(freq_count[i]-predict[i])/predict[i]
    X2=sum(X)'''
    x = range(9)
    X2,predict,k=poisson_test(x,freq_count,b_itr)
    X2_2015,predict_2015,k_2015=poisson_test(x,freq_count_2015,b_itr_2015)
    X2_2016,predict_2016,k_2016=poisson_test(x,freq_count_2016,b_itr_2016)

    #(all expectation greater than 1
    # 80% present of the expectation greater than 5
    print('Chi-squared test, 2017 X^2 value is: ',X2)
    print('prediction poisson distribution k_2017= ', k)
    print('Chi-squared test, 2015 X^2 value is: ', X2_2015)
    print('prediction poisson distribution k_2015= ', k_2015)
    print('Chi-squared test, 2016 X^2 value is: ', X2_2016)
    print('prediction poisson distribution k_2016= ', k_2016)
    predict_sum = CDF(k, b_itr)
    predict_2015_sum = CDF(k_2015, b_itr_2015)
    predict_2016_sum = CDF(k_2016, b_itr_2016)
    # how to find chi squared p?
    plot_data(1,'2017',x,predict,freq_count)
    plot_data(2,'2015',x,predict_2015,freq_count_2015)
    plot_data(3,'2016',x,predict_2016,freq_count_2016)
    np.savetxt('freq_2015.csv',freq_count_2015,fmt="%d",delimiter =',')
    np.savetxt('freq_2016.csv',freq_count_2016,fmt="%d",delimiter =',')
    np.savetxt('freq_2017.csv',freq_count,fmt="%d",delimiter =',')
    L,U=find_CI(b_itr,sum_b)
    L_2015,U_2015=find_CI(b_itr_2015,predict_2015_sum)
    L_2016, U_2016 = find_CI(b_itr_2016, predict_2016_sum)
    #plot_cdf_predict(predict_sum,L,U,'2017',b_itr,'r')
    plot_cdf_predict(predict_2015_sum, L_2015, U_2015, '2015', b_itr_2015,'r')
    plot_cdf_predict(predict_2016_sum, L_2016, U_2016, '2016', b_itr_2016,'b')
    plot_cdf_predict(sum_b, L, U, '2017', b_itr, 'g')
    #Instructions:
    # b,b_2015,b_2016 is the shooting happened each day, wrt yeat 2017,2015,2016
    #each starting from Jan.1
    #plz use the debug mode to run to the end line of 'Pass' to see the values
    # Dependencies: numpy, scipy, matplotlib
pass