
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.neural_network import MLPRegressor


mse=mae=float(0)

def regrevaluate(t,predict, criterion):
    if criterion == 'mse':
        value=np.mean((np.subtract(t,predict))**2)
    else:
        value=np.mean(np.abs(np.subtract(t,predict)))
    return value

#read from the file
boston_dataset = load_boston()

data=boston_dataset['data']

NumberOfAttributes=len(data[0,:])
NumberOfPatterns=len(data)


#initialize
x=data
t=boston_dataset['target']

#num of neuros 
N=[5,10,20,30,40,50,100]

solver=['ibfgs','sgd','adam']
#start_Of_folds
fig,subplt=plt.subplots(3,3);
    
n_folds=9;
for folds in range(0,n_folds):
    
    xtrain,xtest,ttrain,ttest=train_test_split(x,t,test_size=0.25)
    
    numberOfTrain=len(xtrain)
    numberOfTest=len(xtest)
        
    xtrain = np.array(xtrain, dtype=float)
    xtest = np.array(xtest, dtype=float)
    
    model=MLPRegressor(hidden_layer_sizes=N[6],activation='relu',solver='adam')
    model.fit(xtrain,ttrain)
    predict=model.predict(xtest)
    
    mse+=regrevaluate(ttest,predict,'mse')
    mae+=regrevaluate(ttest,predict,'msa')
    
    #plots
    subplt[(folds)/3, (folds)%3].plot(ttest, "ro")
    subplt[(folds)/3, (folds)%3].plot(predict, "b.")

print('Mean Squared Error for all folds is : %f\n',np.mean(mse))
print('Mean Absolute Error for all folds is : %f\n',np.mean(mae))
print('\n');

