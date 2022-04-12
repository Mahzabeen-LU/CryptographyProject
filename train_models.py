###############################################
#    GurobiOptimization Toolbox
#    AUTHOR: MAHZABEEN     
# Requirements:
#    1) Python3
#    2) Gurobi Python
#    3) Numpy
#    4) Random
#    5) tensorFlow & tf-encrypted
###############################################

import random
from gurobipy import *
import numpy as np
import gurobipy as grb
import time
import tensorflow as tf
from tensorflow.keras import utils

#SIMULATION PARAMETERS

NUM_VNFS=500
NUM_DC=5

delta_worst=5000000
NUM_eNBs=5
set_I =range(0, NUM_VNFS)
set_J = range(0, NUM_eNBs)
set_K = range(0, NUM_DC)

#SIMULATION PARAMETERS
Relocation_Time=np.array([[[random.randint(3,5) for i in range(NUM_VNFS)] for j in range(NUM_eNBs)] for k in range(NUM_DC)])
Relocation_Cost_Of_DC= {k: random.randint(20,30) for k in set_K}

bool_relocate= np.array([[random.randint(0,1) for i in range(NUM_VNFS)] for k in range(NUM_DC)])

Communication_Delay=np.array([[[random.randint(25,55) for i in range(NUM_VNFS)] for j in range(NUM_eNBs)] for k in range(NUM_DC)])

Communication_Cost_Of_DC= {k: random.randint(10, 60) for k in set_K}

execution_time={i: random.randint(1,5) for i in set_I}
max_capacity={k: random.randint(10,100000) for k in set_K}

gamma=0.7

verbose=True
t1=time.time()
m = grb.Model("vnf")

""" binary solution (variables) """
x_vars  ={(k,j,i):m.addVar(vtype=grb.GRB.BINARY, name="x_{0}_{0}_{0}".format(k,j,i)) 
                        for k in set_K for j in set_J for i in set_I}

m.update()           
""" optimization objective """
obj_function = 0

for k in range(NUM_DC):
    for j in range(NUM_eNBs):
        for i in range(NUM_VNFS):          
            obj_function += (gamma *bool_relocate[k,i]* Relocation_Time[k,j,i] * x_vars[k,j,i] * Relocation_Cost_Of_DC[k]+ (1-gamma)* Communication_Delay[k,j,i]* x_vars[k,j,i] *Communication_Cost_Of_DC[k])
m.setObjective(obj_function, grb.GRB.MINIMIZE)


""" constraint 5 (The Capacity constraint) """
for k in range(NUM_DC):
    capacity=0
    for j in range(NUM_eNBs):
        for i in range(NUM_VNFS):
            capacity += x_vars[k,j,i]
    m.addConstr(capacity <= max_capacity[k])
    
""" constraint 4 (The QoS constraint) """
for i in range(NUM_VNFS):
    for j in range(NUM_eNBs):
        delta = 0
        for k in range(NUM_DC):
            delta +=(bool_relocate[k,i]* Relocation_Time[k,j,i] * x_vars[k,j,i]+Communication_Delay[k,j,i]+execution_time[i])
        m.addConstr(delta <= delta_worst)
    
""" constraint 3 (allocation ensurity) """
for j in range(NUM_eNBs):
    count=0
    for i in range(NUM_VNFS):
        for k in range(NUM_DC):
            count += x_vars[k,j,i]
    m.addConstr(count == NUM_VNFS)

""" constraint 2 (ensure each VNF of every eNB has one matched Data center [atomicity]) """
for i in range(NUM_VNFS):
    for j in range(NUM_eNBs):
        tmp = 0
        for k in range(NUM_DC):
            tmp +=x_vars[k,j,i]
        m.addConstr(tmp == 1)

""" optimize model """
m.optimize()
t2=time.time()

grb_time=t2-t1

#Generation of Labeled Datasets

b_kji=np.array([[[0 for i in range(NUM_VNFS)] for j in range(NUM_eNBs)] for k in range(NUM_DC)])
for k in range(NUM_DC):
    for j in range(NUM_eNBs):
        for i in range(NUM_VNFS): 
            b_kji[k][j][i]=x_vars[k,j,i].X
  
#conversion into dataframe
import pandas as pd          
index = pd.MultiIndex.from_product([range(s)for s in b_kji.shape])
output_df = pd.DataFrame({'b_kji': b_kji.flatten()}, index=index)['b_kji']
output_df = output_df.unstack(level=0).swaplevel().sort_index()

#Input DataFrame
index = pd.MultiIndex.from_product([range(s)for s in Relocation_Time.shape])
input_df = pd.DataFrame({'Relocation_Time': Relocation_Time.flatten()}, index=index)['Relocation_Time']
input_df = input_df.unstack(level=0).swaplevel().sort_index()

list_VNF=[]

########assembling columns into the dataset: VNF configuration for features#################
for i in set_I:
    for j in range(0,NUM_eNBs):
        list_VNF.append(i)
        
input_df["VNF"]=list_VNF

list_eNB=[]

for i in set_I:
    for j in set_J:
        list_eNB.append(j)
        
input_df["eNB"]=list_eNB

for x in set_K:
    list_cost=[]
    for i in range(0, (NUM_VNFS*NUM_eNBs)):
        list_cost.append(Relocation_Cost_Of_DC[x])
    input_df["RC"+str(x)]=list_cost
      

index = pd.MultiIndex.from_product([range(s)for s in Communication_Delay.shape])
c_delay_df = pd.DataFrame({'Communication_Delay': Communication_Delay.flatten()}, index=index)['Communication_Delay']
input_df=pd.concat((input_df,c_delay_df.unstack(level=0).swaplevel().sort_index()), axis=1) 


list_exec=[]
for i in set_I:
    exec_item=execution_time[i]
    for j in set_J:
        list_exec.append(exec_item)
        
input_df["exec_time"]=list_exec


list_bool_reloc=[]

for k in set_K:
    list_bool_reloc=[]
    for i in set_I:
        for j in range(0,NUM_eNBs):
            list_bool_reloc.append(bool_relocate[k][i])
    input_df["bool_reloc"+str(k)]=list_bool_reloc


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()



#Model Training Libraries
from tensorflow import keras 
from keras.models import Sequential
from keras.layers.convolutional import Conv1D, MaxPooling1D

model=Sequential()

#Preparation of train/test sets
a=np.array(input_df)
b=np.array(output_df)


c=np.array(pd.read_csv('testX_500.csv'))
d=np.array(pd.read_csv('testY_500.csv'))

a=a.astype('float32')
b=b.astype('float32')
c=c.astype('float32')
d=d.astype('float32')

#deriving input shapes for ANN
train_row, train_col=a.shape
test_row, test_col=c.shape

a_cnn=a.reshape(train_row, train_col,1)
c_cnn=c.reshape(test_row, test_col,1)
#training parameters
verbose, epochs, batch_size = 0, 10, 2
n_timesteps, n_features, n_outputs = c_cnn.shape[1], c_cnn.shape[2], d.shape[1]


t1=time.time()

############################ANN#########################################

ann_model = Sequential()
ann_model.add(Dense(100, input_dim=a_cnn.shape[1], activation='relu'))
ann_model.add(Dense(100, activation='relu'))
ann_model.add(Dense(5, activation='softmax'))
ann_model.compile(loss='categorical_crossentropy',
           optimizer='adam',
           metrics=['accuracy'])
               
ann_model.fit(c,d,epochs=epochs, batch_size=batch_size, verbose=1)
prediction_ann=ann_model.predict(a)


prediction_ann=pd.DataFrame(prediction_ann)
t2=time.time()

ann_time=t2-t1
 #confidence score to label
for inx in range(0, (NUM_VNFS*NUM_eNBs)):
     sel=np.argmax(prediction_ann.iloc[inx])
     prediction_ann.iloc[inx][sel]=1

##########Encrypted training################################
#importing necessary libraries
import tensorflow as tf
from tensorflow.keras import utils
from tensorflow import keras 
from keras.models import Sequential
from keras.layers.convolutional import Conv1D, MaxPooling1D

import tf_encrypted as tfe
from tf_encrypted.keras import optimizers


import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from tf_encrypted.keras.optimizers import SGD
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

######number of federated clents set as 3###########
clients=3

#Private training

for i in range(1, clients+1):

    prot = tfe.protocol.Pond()
    
    if(i==1):
        c=np.array(pd.read_csv('testX_500.csv'))
        d=np.array(pd.read_csv('testY_500.csv'))
        
    if(i==2):
        c=np.array(pd.read_csv('testX_500_1.csv'))
        d=np.array(pd.read_csv('testY_500_1.csv'))
    
    if(i==3):
        c=np.array(pd.read_csv('testX_500_2.csv'))
        d=np.array(pd.read_csv('testY_500_2.csv'))
        
        
    c=c.astype('float32')
    d=d.astype('float32')
    
    training_set_size = c.shape[0]
    batch_size = 32
    steps_per_epoch = (training_set_size // batch_size)


    c= prot.define_private_variable(c)
    d= prot.define_private_variable(d)
    #a= prot.define_private_variable(a)

    ann_model = tfe.keras.models.Sequential()
    ann_model.add(tfe.keras.layers.Dense(100,  input_shape=(2, 23), activation='sigmoid'))
    ann_model.add(tfe.keras.layers.Dense(100, activation='sigmoid'))
    ann_model.add(tfe.keras.layers.Dense(5, activation ='sigmoid'))
    ann_model.compile(loss= BinaryCrossentropy() , optimizer=SGD(lr=0.5))   

    ann_model.fit(c,d, epochs=1, steps_per_epoch=2)
    #prediction_ann=ann_model(a)
