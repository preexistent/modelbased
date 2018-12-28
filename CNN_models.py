#!/usr/bin/env python
# coding: utf-8

# In[62]:


import keras
from keras.utils import to_categorical
import pandas
# from keras.datasets import mnist
#download mnist data and split into train and test sets
# (X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[63]:


from performancePlot import plotDistance, computeDistance


# In[64]:


# # Model from the paper
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Activation
from keras import regularizers

# Create a sequential model
model1 = Sequential()

# add model layers
model1.add(Conv2D(4, 
                 kernel_size=1, 
                 activation='relu', 
                 input_shape=(8,8,1), 
                 strides=1,
                 kernel_regularizer=regularizers.l2(0.0001)))
model1.add(Conv2D(4, 
                 kernel_size=1, 
                 activation='relu',
                 strides=1,
                 kernel_regularizer=regularizers.l2(0.0001)))
model1.add(Conv2D(8, 
                 kernel_size=1, 
                 activation='relu',
                 strides=1,
                 kernel_regularizer=regularizers.l2(0.0001)))
model1.add(Flatten())
model1.add(Dense(256,
                activation='relu',
                kernel_regularizer=regularizers.l2(0.0001)))
model1.add(Dense(8,
                 activation='softmax',
                kernel_regularizer=regularizers.l2(0.0001)))


# In[65]:


# Create a sequential model
model2 = Sequential()

# add model layers
model2.add(Conv2D(4, 
                 kernel_size=1, 
                 activation='relu', 
                 input_shape=(8,8,1), 
                 strides=1,
                 kernel_regularizer=regularizers.l2(0.0001)))
model2.add(Conv2D(4, 
                 kernel_size=1, 
                 activation='relu',
                 strides=1,
                 kernel_regularizer=regularizers.l2(0.0001)))
model2.add(Conv2D(8, 
                 kernel_size=1, 
                 activation='relu',
                 strides=1,
                 kernel_regularizer=regularizers.l2(0.0001)))
model2.add(Flatten())
model2.add(Dense(256,
                activation='relu',
                kernel_regularizer=regularizers.l2(0.0001)))
model2.add(Dense(8,
                 activation='softmax',
                kernel_regularizer=regularizers.l2(0.0001)))


# In[66]:


# Create a sequential model
model3 = Sequential()

# add model layers
model3.add(Conv2D(4, 
                 kernel_size=1, 
                 activation='relu', 
                 input_shape=(8,8,1), 
                 strides=1,
                 kernel_regularizer=regularizers.l2(0.0001)))
model3.add(Conv2D(4, 
                 kernel_size=1, 
                 activation='relu',
                 strides=1,
                 kernel_regularizer=regularizers.l2(0.0001)))
model3.add(Conv2D(8, 
                 kernel_size=1, 
                 activation='relu',
                 strides=1,
                 kernel_regularizer=regularizers.l2(0.0001)))
model3.add(Flatten())
model3.add(Dense(256,
                activation='relu',
                kernel_regularizer=regularizers.l2(0.0001)))
model3.add(Dense(8,
                 activation='softmax',
                kernel_regularizer=regularizers.l2(0.0001)))


# In[67]:


# Create a sequential model
model4 = Sequential()

# add model layers
model4.add(Conv2D(4, 
                 kernel_size=1, 
                 activation='relu', 
                 input_shape=(8,8,1), 
                 strides=1,
                 kernel_regularizer=regularizers.l2(0.0001)))
model4.add(Conv2D(4, 
                 kernel_size=1, 
                 activation='relu',
                 strides=1,
                 kernel_regularizer=regularizers.l2(0.0001)))
model4.add(Conv2D(8, 
                 kernel_size=1, 
                 activation='relu',
                 strides=1,
                 kernel_regularizer=regularizers.l2(0.0001)))
model4.add(Flatten())
model4.add(Dense(256,
                activation='relu',
                kernel_regularizer=regularizers.l2(0.0001)))
model4.add(Dense(8,
                 activation='softmax',
                kernel_regularizer=regularizers.l2(0.0001)))


# In[68]:


# Create a sequential model
model5 = Sequential()

# add model layers
model5.add(Conv2D(4, 
                 kernel_size=1, 
                 activation='relu', 
                 input_shape=(8,8,1), 
                 strides=1,
                 kernel_regularizer=regularizers.l2(0.0001)))
model5.add(Conv2D(4, 
                 kernel_size=1, 
                 activation='relu',
                 strides=1,
                 kernel_regularizer=regularizers.l2(0.0001)))
model5.add(Conv2D(8, 
                 kernel_size=1, 
                 activation='relu',
                 strides=1,
                 kernel_regularizer=regularizers.l2(0.0001)))
model5.add(Flatten())
model5.add(Dense(256,
                activation='relu',
                kernel_regularizer=regularizers.l2(0.0001)))
model5.add(Dense(8,
                 activation='softmax',
                kernel_regularizer=regularizers.l2(0.0001)))


# In[69]:


# Create a sequential model
model6 = Sequential()

# add model layers
model6.add(Conv2D(4, 
                 kernel_size=1, 
                 activation='relu', 
                 input_shape=(8,8,1), 
                 strides=1,
                 kernel_regularizer=regularizers.l2(0.0001)))
model6.add(Conv2D(4, 
                 kernel_size=1, 
                 activation='relu',
                 strides=1,
                 kernel_regularizer=regularizers.l2(0.0001)))
model6.add(Conv2D(8, 
                 kernel_size=1, 
                 activation='relu',
                 strides=1,
                 kernel_regularizer=regularizers.l2(0.0001)))
model6.add(Flatten())
model6.add(Dense(256,
                activation='relu',
                kernel_regularizer=regularizers.l2(0.0001)))
model6.add(Dense(8,
                 activation='softmax',
                kernel_regularizer=regularizers.l2(0.0001)))


# In[70]:


# Create a sequential model
model8 = Sequential()

# add model layers
model8.add(Conv2D(4, 
                 kernel_size=1, 
                 activation='relu', 
                 input_shape=(8,8,1), 
                 strides=1,
                 kernel_regularizer=regularizers.l2(0.0001)))
model8.add(Conv2D(4, 
                 kernel_size=1, 
                 activation='relu',
                 strides=1,
                 kernel_regularizer=regularizers.l2(0.0001)))
model8.add(Conv2D(8, 
                 kernel_size=1, 
                 activation='relu',
                 strides=1,
                 kernel_regularizer=regularizers.l2(0.0001)))
model8.add(Flatten())
model8.add(Dense(256,
                activation='relu',
                kernel_regularizer=regularizers.l2(0.0001)))
model8.add(Dense(8,
                 activation='softmax',
                kernel_regularizer=regularizers.l2(0.0001)))


# In[71]:


# Create a sequential model
model7 = Sequential()

# add model layers
model7.add(Conv2D(4, 
                 kernel_size=1, 
                 activation='relu', 
                 input_shape=(8,8,1), 
                 strides=1,
                 kernel_regularizer=regularizers.l2(0.0001)))
model7.add(Conv2D(4, 
                 kernel_size=1, 
                 activation='relu',
                 strides=1,
                 kernel_regularizer=regularizers.l2(0.0001)))
model7.add(Conv2D(8, 
                 kernel_size=1, 
                 activation='relu',
                 strides=1,
                 kernel_regularizer=regularizers.l2(0.0001)))
model7.add(Flatten())
model7.add(Dense(256,
                activation='relu',
                kernel_regularizer=regularizers.l2(0.0001)))
model7.add(Dense(8,
                 activation='softmax',
                kernel_regularizer=regularizers.l2(0.0001)))


# In[72]:


#compile model using accuracy to measure model performance
adam = keras.optimizers.Adam(lr=0.00001,)
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model4.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model5.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model6.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model7.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model8.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[73]:


import numpy as np


# In[74]:


#distanceMatrices = np.loadtxt('distanceMatrices.csv',dtype=float)
#assignmentMatrices = np.loadtxt('assignmentMatrices.csv',dtype=int)
distanceMatrices = np.genfromtxt('distanceMatrices.csv',dtype=float,max_rows=440000)
assignmentMatrices = np.genfromtxt('assignmentMatrices.csv',dtype=int,max_rows=440000)


# In[75]:


X_test = distanceMatrices[400000:420000,]
y_test = assignmentMatrices[400000:420000,]
distanceMatrices = distanceMatrices[:400000,]
assignmentMatrices = assignmentMatrices[:400000,]


# In[76]:


distanceMatrices.shape


# In[77]:


# y_train = to_categorical(y_train)
N,M = assignmentMatrices.shape

# Create a MxNxM matrices,within which matrices[i,:,:] is the ground truth for model i
assignment_onehot = np.zeros((M, N, M))
for i in range(M):
    assignment_onehot[i,:,:] = to_categorical(assignmentMatrices[:,i],num_classes=M)


# In[78]:


assignment_onehot.shape


# In[79]:


N, M = distanceMatrices.shape
distanceMatrices = distanceMatrices.reshape((N,8,8,1))


# In[80]:


assignment_onehot.shape


# In[81]:


recover_assignment = np.zeros_like(assignmentMatrices)
recover_assignment[:,0] = np.argmax(assignment_onehot[0,],axis=1)
recover_assignment[:,1] = np.argmax(assignment_onehot[1,],axis=1)
recover_assignment[:,2] = np.argmax(assignment_onehot[2,],axis=1)
recover_assignment[:,3] = np.argmax(assignment_onehot[3,],axis=1)


# In[82]:


NTrain = 350000
X_train = distanceMatrices[:NTrain,]
X_valid = distanceMatrices[NTrain:,]
y_train_1 = assignment_onehot[0,:NTrain,:]
y_valid_1 = assignment_onehot[0,NTrain:,:]
y_train_2 = assignment_onehot[1,:NTrain,:]
y_valid_2 = assignment_onehot[1,NTrain:,:]
y_train_3 = assignment_onehot[2,:NTrain,:]
y_valid_3 = assignment_onehot[2,NTrain:,:]
y_train_4 = assignment_onehot[3,:NTrain,:]
y_valid_4 = assignment_onehot[3,NTrain:,:]
y_train_5 = assignment_onehot[4,:NTrain,:]
y_valid_5 = assignment_onehot[4,NTrain:,:]
y_train_6 = assignment_onehot[5,:NTrain,:]
y_valid_6 = assignment_onehot[5,NTrain:,:]
y_train_7 = assignment_onehot[6,:NTrain,:]
y_valid_7 = assignment_onehot[6,NTrain:,:]
y_train_8 = assignment_onehot[7,:NTrain,:]
y_valid_8 = assignment_onehot[7,NTrain:,:]


# In[83]:


y_train_1.shape

X_test = np.loadtxt('distanceMatricesTest.csv',dtype=float)
y_test = np.loadtxt('assignmentMatricesTest.csv',dtype=int)
# In[84]:


# y_test = to_categorical(y_test)
N,M = y_test.shape

# Create a MxNxM matrices,within which matrices[i,:,:] is the ground truth for model i
assignment_onehot = np.zeros((M, N, M))
for i in range(M):
    assignment_onehot[i,:,:] = to_categorical(y_test[:,i],num_classes=M)


# In[85]:


N, M = X_test.shape
X_test = X_test.reshape((N,8,8,1))


# In[86]:


epochs = 10

res_train = []
res_test = []


# In[87]:


y_train_1.shape


# In[88]:


for i in range(5):
    model1.fit(X_train, y_train_1, batch_size=1000, validation_data=(X_valid, y_valid_1), epochs=epochs)
    model2.fit(X_train, y_train_2, batch_size=1000, validation_data=(X_valid, y_valid_2), epochs=epochs)
    model3.fit(X_train, y_train_3, batch_size=1000, validation_data=(X_valid, y_valid_3), epochs=epochs)
    model4.fit(X_train, y_train_4, batch_size=1000, validation_data=(X_valid, y_valid_4), epochs=epochs)
    model5.fit(X_train, y_train_5, batch_size=1000, validation_data=(X_valid, y_valid_5), epochs=epochs)
    model6.fit(X_train, y_train_6, batch_size=1000, validation_data=(X_valid, y_valid_6), epochs=epochs)
    model7.fit(X_train, y_train_7, batch_size=1000, validation_data=(X_valid, y_valid_7), epochs=epochs)
    model8.fit(X_train, y_train_8, batch_size=1000, validation_data=(X_valid, y_valid_8), epochs=epochs)
    
    predicted_train_1 = model1.predict(X_train)
    predicted_train_2 = model2.predict(X_train)
    predicted_train_3 = model3.predict(X_train)
    predicted_train_4 = model4.predict(X_train)
    predicted_train_5 = model5.predict(X_train)
    predicted_train_6 = model6.predict(X_train)
    predicted_train_7 = model7.predict(X_train)
    predicted_train_8 = model8.predict(X_train)
    
    predicted_test_1 = model1.predict(X_test)
    predicted_test_2 = model2.predict(X_test)
    predicted_test_3 = model3.predict(X_test)
    predicted_test_4 = model4.predict(X_test)
    predicted_test_5 = model5.predict(X_test)
    predicted_test_6 = model6.predict(X_test)
    predicted_test_7 = model7.predict(X_test)
    predicted_test_8 = model8.predict(X_test)
    
    recover_train = np.zeros_like(predicted_train_1)
    recover_train[:,0] = np.argmax(predicted_train_1,axis=1)
    recover_train[:,1] = np.argmax(predicted_train_2,axis=1)
    recover_train[:,2] = np.argmax(predicted_train_3,axis=1)
    recover_train[:,3] = np.argmax(predicted_train_4,axis=1)
    recover_train[:,4] = np.argmax(predicted_train_5,axis=1)
    recover_train[:,5] = np.argmax(predicted_train_6,axis=1)
    recover_train[:,6] = np.argmax(predicted_train_7,axis=1)
    recover_train[:,7] = np.argmax(predicted_train_8,axis=1)
    
    recover_test = np.zeros_like(predicted_test_1)
    recover_test[:,0] = np.argmax(predicted_test_1,axis=1)
    recover_test[:,1] = np.argmax(predicted_test_2,axis=1)
    recover_test[:,2] = np.argmax(predicted_test_3,axis=1)
    recover_test[:,3] = np.argmax(predicted_test_4,axis=1)
    recover_test[:,4] = np.argmax(predicted_test_5,axis=1)
    recover_test[:,5] = np.argmax(predicted_test_6,axis=1)
    recover_test[:,6] = np.argmax(predicted_test_7,axis=1)
    recover_test[:,7] = np.argmax(predicted_test_8,axis=1)
    
    distance_train = computeDistance(X_train, recover_train)
    distance_test = computeDistance(X_test, recover_test)
    res_train.append(distance_train)
    res_test.append(distance_test)


# In[40]:


res_train
np.savetxt('./csv_8x8/train_distance.csv',res_train,delimiter=',',fmt='%f')


# In[41]:


res_test
np.savetxt('./csv_8x8/test_distance.csv',res_test,delimiter=',',fmt='%f')


# In[110]:


optimal_Distance_train = computeDistance(X_train, assignmentMatrices[:NTrain,])
optimal_Distance_test = computeDistance(X_test, y_test)
np.savetxt('./csv_8x8/train_optimal_distance.csv',[optimal_Distance_train]*100,delimiter=',',fmt='%f')
np.savetxt('./csv_8x8/test_optimal_distance.csv',[optimal_Distance_test]*100,delimiter=',',fmt='%f')


# # Save the model

# In[ ]:


model1_json = model1.to_json()
model2_json = model1.to_json()
model3_json = model1.to_json()
model4_json = model1.to_json()
model5_json = model1.to_json()
model6_json = model1.to_json()
model7_json = model1.to_json()
model8_json = model1.to_json()
with open('model1.json','w') as json_file:
    json_file.write(model1_json)
with open('model2.json','w') as json_file:
    json_file.write(model2_json)
with open('model3.json','w') as json_file:
    json_file.write(model3_json)
with open('model4.json','w') as json_file:
    json_file.write(model4_json)
with open('model5.json','w') as json_file:
    json_file.write(model5_json)
with open('model6.json','w') as json_file:
    json_file.write(model6_json)
with open('model7.json','w') as json_file:
    json_file.write(model7_json)
with open('model8.json','w') as json_file:
    json_file.write(model8_json)
model1.save_weights('model1.h5')
model2.save_weights('model2.h5')
model3.save_weights('model3.h5')
model4.save_weights('model4.h5')
model5.save_weights('model5.h5')
model6.save_weights('model6.h5')
model7.save_weights('model7.h5')
model8.save_weights('model8.h5')
print('Saved model to disk')


# # Load the model

# In[6]:


from keras.models import model_from_json
json_file = open('model1.json', 'r') 
model1_json = json_file.read() 
json_file.close() 
model1 = model_from_json(model1_json)
model1.load_weights("model1.h5") 
print("Loaded model from disk")

json_file = open('model2.json', 'r') 
model2_json = json_file.read() 
json_file.close() 
model2 = model_from_json(model2_json)
model2.load_weights("model2.h5") 
print("Loaded model from disk")

json_file = open('model3.json', 'r') 
model3_json = json_file.read() 
json_file.close() 
model3 = model_from_json(model3_json)
model3.load_weights("model3.h5") 
print("Loaded model from disk")

json_file = open('model4.json', 'r') 
model4_json = json_file.read() 
json_file.close() 
model4 = model_from_json(model4_json)
model4.load_weights("model4.h5") 
print("Loaded model from disk")

json_file = open('model5.json', 'r') 
model5_json = json_file.read() 
json_file.close() 
model5 = model_from_json(model5_json)
model5.load_weights("model5.h5") 
print("Loaded model from disk")

json_file = open('model6.json', 'r') 
model6_json = json_file.read() 
json_file.close() 
model6 = model_from_json(model6_json)
model6.load_weights("model6.h5") 
print("Loaded model from disk")

json_file = open('model7.json', 'r') 
model7_json = json_file.read() 
json_file.close() 
model7 = model_from_json(model7_json)
model7.load_weights("model4.h5") 
print("Loaded model from disk")

json_file = open('model8.json', 'r') 
model8_json = json_file.read() 
json_file.close() 
model8 = model_from_json(model_json)
model8.load_weights("model8.h5") 
print("Loaded model from disk")

