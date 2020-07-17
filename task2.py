# Importing all libraries required in this notebook
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt 
import pickle 
#%matplotlib inline
df = pd.read_csv('student_scores - student_scores.csv')
print("Data imported successfully")
df.head()
#Let's plot our data points on 2-D graph to eyeball our dataset and see if we can manually find any relationship between the data. We can create the plot with the following script:

# Plotting the distribution of scores
df.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()

#From the graph above, we can clearly see that there is a positive linear relation between the number of hours studied and percentage of score.

#Preparing the data
X = df.iloc[:, :-1].values 
#X = X.reshape(-1,1)
y = df.iloc[:, 1].values 
#y = y.reshape(-1,1)
#Now that we have our attributes and labels, the next step is to split this data into training and test sets. We'll do this by using Scikit-Learn's built-in train_test_split() method:

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 
#Training the Algorithm
from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 
print("Training complete.")
#Training complete.
# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_
plt.scatter(X, y)
plt.plot(X, line);
plt.show()
#Making Predictions
#Now that we have trained our algorithm, it's time to make some predictions.
print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores
# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 
pred = regressor.predict(X_test)
plt.scatter(y_test,pred)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
#Looks like our model is working well. So lets try to give a new input and predict.
#pred = regressor.predict([[9.5]])
#print("Predicted Score = {}".format(pred[0]))
#Evaluating the model
#The final step is to evaluate the performance of algorithm. This step is particularly important to compare how well different algorithms perform on a particular dataset.
from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 
print('Mean Squared Error:', metrics.mean_squared_error(y_test, pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, pred)))
pickle.dump(regressor,open('test2.pkl','wb'))
model=pickle.load(open('test2.pkl','rb'))