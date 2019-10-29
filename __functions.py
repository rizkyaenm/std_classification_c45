#!/usr/bin/env python
# coding: utf-8

# # Functions

# ## Mean Imputation 

# In[1]:


from __setup import *

'''function printmd()
funtion to print string in using markdown and display library

input:
    string

output:
    string
'''


def printmd(string):
    display(Markdown(string))
    
'''
'''
### function : mean, mode 
#random_mean_class
def mean_missdata(data, cols, cls):
    
    col_logs =['atribut','kelas', 'mean','std','q1','q3']
    
    log_mean =  pd.DataFrame(columns =col_logs )
    newdata = data.copy(deep=True)
    length = len(newdata)
    df_rands = pd.DataFrame(columns=cols, index=range(0,length))
    
    class_data = newdata[cls].value_counts().index.tolist()
    newdata.rename(columns={cls:'Class'}, inplace=True)
    combine = list(itertools.product(cols, class_data))
    print(combine)
    
    for column, class_val in combine:

#         if data[column][newdata.Class == class_val].isnull().any() :
        if data[column][newdata.Class == class_val].isnull().any() :
            
            subset = newdata[column][newdata.Class == class_val]
            print ("subset", subset)
     
            checksub=subset.isnull()
            print ("checksub", checksub)
            sub_null = checksub[checksub == True]
            
            indexsub = sub_null.index.values
            print( indexsub)
                
            std = subset.std()
            mean = subset.mean()
            start = mean - std
            end = mean + std
#             start=start.round(4)
#             end =end.round(4)
            
        
            df_rands[column]= np.random.uniform(start,end,length)
        
            
            newdata.at[indexsub, column]=df_rands[column]
        
            df_drands = df_rands.drop([column], axis=1)
            log_mean =log_mean.append({'atribut': column, 'kelas': class_val, 'mean':mean,
                                      'std':std, 'q1':start,'q3':end
                                      },ignore_index=True)
           
#                
               
    newdata[cols] = newdata[cols].round(2)
    newdata.rename(columns={'Class':cls}, inplace=True)
    display(log_mean)
    
#     log_mean.to_csv('Data Result csv/[3]Tabellog_mean.csv')
    
    return newdata,log_mean


def mean_missdata_row(data, cols, cls):
    
    col_logs =['atribut','kelas', 'mean','std','q1','q3']
    
    log_mean =  pd.DataFrame(columns =col_logs )
    newdata = data.copy(deep=True)
    length = len(newdata)
    df_rands = pd.DataFrame(columns=cols, index=range(0,length))
    
    class_data = newdata[cls].value_counts().index.tolist()
    newdata.rename(columns={cls:'Class'}, inplace=True)
    combine = list(itertools.product(cols, class_data))
    print(combine)
    
    for column, class_val in combine:

#         if data[column][newdata.Class == class_val].isnull().any() :
        if data[column][newdata.Class == class_val].isnull().any() :
            
            subset = newdata[column][newdata.Class == class_val]
            print ("subset", subset)
     
            checksub=subset.isnull()
            print ("checksub", checksub)
            sub_null = checksub[checksub == True]
            
            indexsub = sub_null.index.values
            print( indexsub)
                
            std = subset.std()
            mean = subset.mean()
            start = mean - std
            end = mean + std
#             start=start.round(4)
#             end =end.round(4)
            
        
            df_rands[column]= np.random.uniform(start,end,length)
        
            
            newdata.at[indexsub, column]=df_rands[column]
        
            df_drands = df_rands.drop([column], axis=1)
            log_mean =log_mean.append({'atribut': column, 'kelas': class_val, 'mean':mean,
                                      'std':std, 'q1':start,'q3':end
                                      },ignore_index=True)
           
#                
               
    newdata[cols] = newdata[cols].round(2)
    newdata.rename(columns={'Class':cls}, inplace=True)
    display(log_mean)
    
#     log_mean.to_csv('Data Result csv/[3]Tabellog_mean.csv')


# ## Mode Imputation

# In[2]:


def mode_missdata(data, cols, cls):
    
    cols_log = ['atribut','kelas', 'mode']
    log_mode = pd.DataFrame(columns=cols_log)
    
    
    class_data = data[cls].value_counts().index.tolist()
    data.rename(columns={cls:'Class'}, inplace=True)
    combine = list(itertools.product(cols, class_data))
    
    for column, class_val in combine:
        
        if data[column][data.Class == class_val].isnull().any() :
            
            subset = data[column][data.Class == class_val]
            checksub=subset.isnull()
            sub_null = checksub[checksub==True]
            indexsub = sub_null.index.values
            
            
            mode = data[column][data.Class == class_val].mode().iloc[0]
#             print "mode", dt_rands
            data[column][data.Class == class_val] = data[column][data.Class == class_val].fillna(mode)
            log_mode =log_mode.append({'atribut': column, 'kelas': class_val, 'mode':mode},
                                      ignore_index=True)
    
    
    
    data.rename(columns={'Class':cls}, inplace=True) 
    display(log_mode)
#     log_mode.to_csv('Data Result csv/log_mode.csv')
    return data,log_mode


# ## Value to Category

# In[3]:


def ip_tocategory(value, feat):
    if value <=4.00 :
         #original : CUM
        if value > 3.50 :
#         if value > 3.25 :
            category = feat + 'CUM'
        else:
            #original SM
            if value > 3.00 :
                category = feat + 'SM'
            else:
                #original :
                if value > 2.75 :
                    category = feat + 'M'
                else:
                    #original : KM
                    if value >= 0.00:
                        category = feat + 'KM'
    else: 
        category =''
                                          
    return category


def sks_tocategory(value, feat):
    if value <=24.00 :
        if value >= 22.00 :
            category = feat + 'SB'
        else:
            if value >= 18.00 :
                category = feat + 'B'
            else:
                if value >= 15.00 :
                    category = feat + 'C'
                else:
                    if value >=12.00 :
                        category = feat + 'K'
                    else :
                        if value < 12.00 :
                            category = feat + 'K'
    else: 
        category =''                     
    
    return category
    


# ## Status_graduation 

# In[4]:



#define kategori status lulus based on angkatan, status lulus mhs dan tahun kelulusan
def status_grad(row):
    stat = 'BL'
    value = row['SMT_IPK_TERAKHIR']
    year_grad, sem = value.split('-')
    year_class = row['ANGKATAN']
    time_grad = int(year_grad) - year_class
    stat_mhs = row['STATUS_MHS']

    if stat_mhs == 'L':
        if time_grad == 3 :
            if sem == 'B':
                stat = 'LTW'
            if sem =='b':
                stat = 'LTW'
            if sem == 'c':
                stat ='LTW'
            if sem == 'C':
                stat == 'LTW'
                
    ##LC original
        if time_grad == 3:
            if sem == 'A':
                stat = 'LC'
            if sem == 'a':
                stat ='LC'
        if time_grad == 4:
            stat = 'LLW_1'
        if time_grad > 4:
            stat = 'LLW_2'
    else :
        stat = 'BL'
    return stat



# ## Categories and prodi

# In[5]:



#variabel insert 'JUR' based on 'NIM'
insert ={
  r'.*M01.*' : 'MTK',
  r'.*M02.*' : 'FISIKA',
  r'.*M04.*' : 'BIO',
  r'.*M05.*' : 'IF',
  r'.*F03.*' : 'AKU',
  r'.*F01.*' : 'EKO',
  r'.*E00.*' : 'HUK',
  r'.*D04.*' : 'HI',
  r'.*D02.*' : 'KOM'

}

Kategori_old = {"PEK_AYAH":{"Pedagang/Wiraswasta":"PDG",
                            "Guru/Dosen Negeri" :"GRD",
                            "Pegawai Negri bukan Guru/Dosen":"PNS",
                            "Pegawai Swasta bukan Guru/Dosen":"PS",
                            "Buruh/Orang yang bekerja dengan tenaga fisik saja":"BPN",
                            "Lain-lain":"LL",
                            "ABRI":"LL",
                            "Petani/Nelayan":"BPN",
                            "Pensiunan Pegawai Negeri/ABRI":"PSN",
                            "Guru/Dosen Swasta":"GRD",
                            "Pensiunan Pegawai Swasta":"PSN",
                            "Ahli Profesional yang hanya bekerja secara perorangan":"LL"},

                  "PEK_IBU":{"Pedagang/Wiraswasta":"PDG",
                            "Guru/Dosen Negeri" :"GRD",
                            "Pegawai Negri bukan Guru/Dosen":"PNS",
                            "Pegawai Swasta bukan Guru/Dosen":"PS",
                            "Buruh/Orang yang bekerja dengan tenaga fisik saja":"BPN",
                            "Lain-lain":"LL",
                            "ABRI":"LL",
                            "Petani/Nelayan":"BPN",
                            "Pensiunan Pegawai Negeri/ABRI":"PSN",
                            "Guru/Dosen Swasta":"GRD",
                            "Pensiunan Pegawai Swasta":"PSN",
                            "Ahli Profesional yang hanya bekerja secara perorangan":"LL"},
                
                "GAJI_ORTU": {"Rp. 1.000.001,00 s.d Rp. 2.500.000,00":"G3",
                             "Rp. 500.000,00 s.d Rp. 1.000.000,00":"G2",
                             "Rp. 2.500.001,00 s.d Rp. 5.000.000,00":"G4",
                             "< Rp. 500.000,00":"G1",
                             "Rp. 5.000.001,00 s.d Rp. 7.500.000,00":"G5",
                             "Rp. 7.500.001,00 s.d Rp. 10.000.000,00":"G6",
                             "Lebih dari Rp. 10.000.000,00":"G7"},

                "PDK_IBU": {"Tamat SMTA":"SMTA",
                            "Sarjana (S1)":"SRJN",
                            "Tamat SD":"SD",
                            "Tamat SMP":"SMP",
                            "Sarjana Muda":"SRJN",
                            "Tidak Tamat SD":"LL",
                            "Pasca Sarjana (S2)":"SRJN",
                            "Lain":"LL",
                            "Pasca Sarjana (S3)":"SRJN"},
              
                }


# ## Handle Duplicate data

# In[6]:


def del_duplicate(data, cols, col_cls ,cls):
    newdata = data
    data_cls = newdata[newdata[col_cls] == cls]
    data_cls = data_cls.drop_duplicates(subset=cols, keep='first', inplace=False)
    return data_cls


# ## divide testing data

# In[7]:


import random
import random
def divide_testing_data(data,cls,cols):
    
    filtered_data = data.copy(deep=True)
    print("filtered data",len(filtered_data))
    filtered_data = filtered_data[cols].dropna()
    print("filtered data dropna",len(filtered_data))
    display(filtered_data)

    list_class = filtered_data[cls].value_counts().index.tolist()
    filtered_data.rename(columns={cls:'Class'}, inplace=True)
    temp1 =pd.DataFrame(columns = filtered_data.columns)
    
    #temp2 =pd.DataFrame(columns = filtered_data.columns)
    
    for class_val in list_class:
        subset = filtered_data[filtered_data.Class == class_val]
    
        
        length = len(subset)
        #half = length/2
#         print ("length subset: ",length)
#         print ("subset", class_val)
        
        list_id1= list(range(0,length))
        #list_id2= list(range(half,length))
        
        number_ofsample = length / 4
        if number_ofsample == 0 :
            number_ofsample = length / 3
        #print ("num. of sample: ", number_ofsample)
        
        getrandom1 = random.sample(list_id1,number_ofsample)
        #getrandom2 = random.sample(list_id2,number_ofsample)
        
        temp1 = temp1.append(subset.iloc[getrandom1,:])
        #temp2 = temp2.append(subset.iloc[getrandom2,:])
#         print(len(temp1))
    
        
    display(temp1.head())
    indexdata1 = temp1.index.values
    print("length data uji: ", len(indexdata1))
    
    #indexdata2 = temp2.index.values
    #print("length data uji: ",len(indexdata2))
    
    temp1.rename(columns={'Class':cls}, inplace=True)
    #temp2.rename(columns={'Class':cls}, inplace=True)
    
    data_1 = data.copy(deep=True)
    #data_2 = data.copy(deep=True)
    
    data_1 = data.drop(indexdata1)
    #data_2 = data.drop(indexdata2)

#    
#     print('data1')
#     display(data_1)
#     print('data uji')
#     display(temp1)
    
    
    return data_1,temp1

# ## Divide testing data Random

# In[8]:


import random
def divide_testing_data_random(data,cls,cols):

    filtered_data = data.copy(deep=True)
    filtered_data = filtered_data[cols].dropna()
    print ("len data sebelum data random",len(filtered_data))
    #list_class = filtered_data[cls].value_counts().index.tolist()
    filtered_data.rename(columns={cls:'Class'}, inplace=True)
    temp = pd.DataFrame(columns=filtered_data.columns)
    length = len(filtered_data)
    list_id= list(range(0,length))
    number_ofsample = length / 4
    getrandom = random.sample(list_id,number_ofsample)
    temp = temp.append(filtered_data.iloc[getrandom,:])
  

    indexdata = temp.index.values
    print("length data uji: ",len(indexdata))

    
    temp.rename(columns={'Class':cls}, inplace=True)
    print "divide random : index data", indexdata 
    
    data = data.drop(indexdata)
    print ("len data setelah data random",len(data))
    
    print sum(data.index.isin(indexdata))
    return data, temp
    


# ## Mean IP_S1 & IP_S2

# In[9]:


def mean_ip(row):
    return float((row['IP_S1'] + row['IP_S2'])/2)


# ## Mean imputation based on individually score

# In[10]:



########################  Function Mean_missdata_rows  ########################

''' 
Function Mean_missdata_rows to handling missing data with calculating 
the value of mean based on student performance from another semester  

params:
    cols (list) - column who needed from the parameters - list            
    row (dataframe/series) - attr. to save one row from data              
    max_attr (int) - total attributes of IPS / SKS                        
    row_isnull (series/dataframe) - the result row after using isnull()   
    isnull()(bool) -  check if the data is null                           
    result(bool) - result from check_null() function 
    sum_null (int) - total of column which have missing value - int    
    mean (float) - mean calculation from columns that filled    
    total_filled_row (float) is total value of filled columns/parameter in rows   

returns:
    row(dataframe)
'''



########################  Function Mean_missdata_rows  ########################

''' 
Function Mean_missdata_rows to handling missing data with calculating 
the value of mean based on student performance from another semester  

params:
    cols (list) - column who needed from the parameters - list            
    row (dataframe/series) - attr. to save one row from data              
    max_attr (int) - total attributes of IPS / SKS                        
    row_isnull (series/dataframe) - the result row after using isnull()   
    isnull()(bool) -  check if the data is null                           
    result(bool) - result from check_null() function 
    sum_null (int) - total of column which have missing value - int    
    mean (float) - mean calculation from columns that filled    
    total_filled_row (float) is total value of filled columns/parameter in rows   

returns:
    row(dataframe)
'''


def mean_missdata_row(cols, row, max_attr, targets): 
    print("1",row[cols].tolist())
    row_isnull = row.isnull()
    sum_null = row_isnull[cols].sum()
    total_filled_row = row[cols].sum(skipna = True, axis = 0)
    mean = total_filled_row /(max_attr - sum_null)
    
    for target in cols:
        result = row_isnull[target]
        print (result )
        if result == True :
            row[target] = mean
            print( 'xxxxx')
    print("2",row[cols].tolist())
    print("-----------------------------------")
            
    return row

def mean_imputation_data(data, cols, max_attr, targets):
    data = data.reset_index(drop=True)
    print "before\n",data
    data = data.dropna(thresh = 1, subset=cols)
    print data
    data = data.reset_index(drop=True)
    
    list_index = pd.isnull(data[targets]).any(1).nonzero()[0]
    print (list_index)
    for index in list_index:
        row = data[cols].iloc[index,:]
        print (row)
        row_isnull = row.isnull()
        sum_null = row_isnull[cols].sum()
        total_filled_row = row[cols].sum(skipna = True, axis = 0)
        mean = total_filled_row /(max_attr - sum_null)
        row.fillna(mean, inplace=True)
        
        data.loc[index,cols] = row
        
        print ("after",row)
        print (data.loc[index,cols])
        
    list_index = pd.isnull(data[targets]).any(1).nonzero()[0]
    print list_index

    return data
        
   
#     log_mean.to_csv('Data Result csv/[3]Tabellog_mean.csv')



def load_data(data_train, data_testing):
    
    loader = Loader(classname="weka.core.converters.ArffLoader")
    train = loader.load_file(data_train + ".arff")
    testing = loader.load_file(data_testing + ".arff")
    
    train.class_is_last()
    testing.class_is_last()
    train.class_index = train.num_attributes - 1
    
    return train,testing

def build_tree(train, testing, directory):
    
    
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.options = ["-U"] 
    cls.build_classifier(train)
    
    # save classifier into cls file/weka file
    outfile = directory + "Unpruned_j48.model"
    serialization.write(outfile, cls)
    f = open( directory + "Unpruned_tree.txt", "w+" )
    f.write('{}'.format(cls))
    f.close()
    
    #save tree classifier into image
    graph.plot_dot_graph(cls.graph, filename = directory + "Unpruned_Result_tree.png")

    return cls


def testing_evaluation(cls, train, testing, directory):
    
    list_summ = []
    evaluation = Evaluation(train, cost_matrix = None)
    evaluation.test_model(cls, testing)
#     print(evaluation.confusion_matrix)

    print (directory)
    print(evaluation.class_details())
    print evaluation.summary(title="Summary", complexity=False)

    # Output predictions
    cols = ['Actual', 'Predicted', 'Error', 'Distribution']
    list_inst =[]
    df =pd.DataFrame(columns=cols)
   
    for index, inst in enumerate(testing):
        pred = cls.classify_instance(inst)
        dist = cls.distribution_for_instance(inst)
        myFormattedList = [ '%.2f' % elem for elem in dist.tolist() ]
        
        

        list_inst.append({'Actual':inst.get_string_value(inst.class_index),
                          'Predicted': inst.class_attribute.value(int(pred)),
                          'Error':"yes" if pred != inst.get_value(inst.class_index)else "no",
                          'Distribution' : myFormattedList,
                         })  

    df = df.append(list_inst)
    df = df.reset_index(drop=True)
    df.to_csv(directory + 'Unpruned_TestingResult.csv', index=False)
    summ = summary(directory, evaluation, list_summ, 0)
    
    
    return df,evaluation,summ

    
    
def summary(directory, evaluation, list_summ, confval):
    
    
    
    sum_item = evaluation.summary().split()
    precision = evaluation.precision(1)  
    class_detail = evaluation.class_details()
    class_detail_item = class_detail.split()
    len_detail = len(class_detail_item)
    sum_item.append(0)
    sum_item.append(confval)
    
    #     precision + recall
 
    
    sum_item.append(class_detail_item[21]+"-"+ class_detail_item[27])
    sum_item.append(class_detail_item[22] + "-"+ class_detail_item[27])

    
    sum_item.append(class_detail_item[30]+ "-"+class_detail_item[36])
    sum_item.append(class_detail_item[31] +"-"+ class_detail_item[36])
    
    sum_item.append(class_detail_item[39] + "-"+class_detail_item[45])
    sum_item.append(class_detail_item[40] +"-"+ class_detail_item[45])
    
    if len_detail == 65 :
        sum_item.append(class_detail_item[48] +"-"+ class_detail_item[54])
        sum_item.append(class_detail_item[49] +"-"+ class_detail_item[54])
#     print (class_detail_item[21],class_detail_item[30], class_detail_item[39])

    
    
    list_summ.append(sum_item)
    
    conf_matrix = evaluation.matrix()
    
    
#     print type(class_detail_item)
#     print class_detail_item 
#     print class_detail_item
    f = open( directory+str(confval)+"Confusion Matrix.txt", "w+" )
    f.write('{} \n {}'.format(conf_matrix, class_detail))
    f.close()
   
    
    return list_summ
    
def pruning_tree(directory, training, testing, list_summ, summary_):

    ######  Recursive Pruning
    for value in range(50,1,-1):

        confval = float(value)/100
        confval = str(confval)
        
        cls = Classifier(classname="weka.classifiers.trees.J48")
        cls.options = ["-C", confval]
        
        
        #### cls.options = ["-R"]
          
#         print "__________Pruning tree with Confidence Value : ", confval
#         print(cls.options)

        
        cls.build_classifier(training)
        
        outfile = directory + confval+ "pruning_j48.model"
        serialization.write(outfile, cls)
        f = open( directory + confval+"_pruning_tree.txt", "w+" )
        f.write('{}'.format(cls))
        f.close()
#         graph.plot_dot_graph(cls.graph, filename = directory +confval+"pruning_j48.png")

        #### Testing
        evaluation = Evaluation(training, cost_matrix = None)
        evaluation.test_model(cls, testing)
        
        summ = summary(directory, evaluation,list_summ,confval)
        
    summary_ = summary_.append(summ)
    summary_ = summary_.reset_index(drop =  True)
    summary_.to_csv(directory +'summ_temp.csv', index=False)
    
    return summary_


def re_summary(summary, directory):
    cols_new = ['Confidence Value','Correctly Classified', 'Incorrectly Classified', 'Kappa', 'MEE',
            'RMSE','Relative abs. error', 'Root Relative abs. error']
    new_summary = pd.DataFrame(columns=cols_new)

    #Assign Summary
    new_summary['Confidence Value'] = summary.iloc[:,41]
    new_summary['precision1'] = summary.iloc[:,42]
    new_summary['recall1'] = summary.iloc[:,43]
    
    new_summary['precision2'] = summary.iloc[:,44]
    new_summary['recall2'] = summary.iloc[:,45]
    
    new_summary['precision3'] = summary.iloc[:,46]
    new_summary['recall3'] = summary.iloc[:,47]
    
    if len(summary.columns) == 50 :
        new_summary['precision4'] = summary.iloc[:,48]
        new_summary['recall4'] = summary.iloc[:,49]

    new_summary['Correctly Classified'] = summary.iloc[:,4]
    new_summary['Incorrectly Classified'] = summary.iloc[:,10]
    new_summary['Kappa'] = summary.iloc[:,14]
    new_summary['MEE'] = summary.iloc[:,18]
    new_summary['RMSE'] = summary.iloc[:,23]
    new_summary['Relative abs. error'] = summary.iloc[:,27]
    new_summary['Root Relative abs. error'] = summary.iloc[:,33]
    
#     new_summary = new_summary[].astype('float64')
    new_summary = new_summary.sort_values(by=['Confidence Value'], ascending=True)

    new_summary = new_summary.reset_index(drop=True)
    new_summary.to_csv( directory + 'Summary.csv')
    return new_summary
#     print display(new_summary)

    
def data_demografis(data,cls):
    datax = pd.DataFrame(columns = data.columns)
    if cls == "1": #IPK
        datax =data['JUR','JK','GAJI_ORTU','PDK_IBU','PEK_IBU','PEK_AYAH','IPK_TERAKHIR']
        datax.to_csv("DataResult/final/uji/data_uji_mipa_ipk_demografis")
    if cls == "2":#STATUS LULUS
        datax = data['JUR','JK','GAJI_ORTU','PDK_IBU','PEK_IBU','PEK_AYAH', 'STATUS_LULUS']
    
