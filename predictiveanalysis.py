import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
def plot_confusion_matrix(y,y_predict):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    cm = confusion_matrix(y, y_predict)
    ax=plt.subplot()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['did not land', 'land'])
    ax.yaxis.set_ticklabels(['did not land', 'land'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

data=pd.read_csv('predictive analysis dataset.csv')
print(data.head())
print(data.info())

X=pd.read_csv('dataset_part_3.csv')
print(X.head(100))

Y=data['Class'].to_numpy()
print(Y)

transform = preprocessing.StandardScaler()
X = transform.fit_transform(X)
print(X[:5])
print(type(X))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
print("Numebr of records in the test dataset: ",len(X_test))
#LOGISTIC REGRESSION ML MODEL
lr=LogisticRegression()
parameters = {'C':[0.01,0.1,1], 'penalty':['l2'], 'solver':['lbfgs']}
#C means inverse of regularization strength . smaller values mean stronger regularization (prevents overfitting)
#penalty is the type of regulariztion, l1 is ridge .. and l2 is lasso.
#solver is the algorithm to use for optimzation.'lbfgs' is good for small/medium datssets and supports l2 regularization

logreg_cv=GridSearchCV(lr,param_grid=parameters,cv=10)
logreg_cv.fit(X_train,Y_train)
print("Best parameters:",logreg_cv.best_params_)
print("Accuracy:",logreg_cv.best_score_)

best_lr=logreg_cv.best_estimator_
accuracy_test=best_lr.score(X_test,Y_test)
print("Test set accuracy of best model:",accuracy_test)

yhat=best_lr.predict(X_test)
print("Predicted values:",yhat)
print("True values:",Y_test)
print("Training set score:", best_lr.score(X_train, Y_train))
print("Test set score:", best_lr.score(X_test, Y_test))
print("Mean Squared Error:", mean_squared_error(Y_test, yhat))
print("R2 Score:", r2_score(Y_test, yhat))
plot_confusion_matrix(Y_test, yhat)


#SVM/SVC CLASSIFIER
parameters2= {'kernel':('linear', 'rbf','poly', 'sigmoid'), #type of the decision boundary the svm will use.
              #linear->straight line(or hyperplane in higher dimensions),rbf->radial basis function(most common,non linear),
              #poly->polynomial kernel,sigmoid->sigmoid kernel.
              'C': np.logspace(-3, 3, 5), #regularization parameter like the one used in logistic regression.
              #smaller c allows misclassifications,encouraging simpler decision boundary.
              #higher c tries to classify training points correctly and can sometimes result in overfitting.
              #c=np.logspace(-3,3,5) generates 5 values evenly spaced on log scale between -10^3 and 10^3
              'gamma':np.logspace(-3, 3, 5)}
#gamma is only relevant for non linear kernels(rbf,poly,sigmoid)
#determines influence of each training point
#lower gamma means far points matter(smooth boundary) and higher gamma means close points matter(wiggly boundary)
#also log spaced values like c.
svm= SVC()
svm_cv=GridSearchCV(svm,param_grid=parameters2,cv=10)
#try all combination of paramaters in the paramaters2.
#using cv=10,10 fold cross vlaidation ...dataset is split into 10 parts(9 for training, 1 for testing)
#repeated 10 times, then averaged.
svm_cv.fit(X_train,Y_train)
print("Best parameters:",svm_cv.best_params_)
print("Accuracy:",svm_cv.best_score_)
best_svm=svm_cv.best_estimator_
accuracy_test_svm=best_svm.score(X_test,Y_test)
print("Test set accuracy of best model:",accuracy_test_svm)
yhat_svm=best_svm.predict(X_test)
print("Predicted values:",yhat_svm)
print("True values:",Y_test)
print("Training set score:", best_svm.score(X_train, Y_train))
print("Test set score:", best_svm.score(X_test, Y_test))
print("Mean Squared Error:", mean_squared_error(Y_test, yhat_svm))
print("R2 Score:", r2_score(Y_test, yhat_svm))
plot_confusion_matrix(Y_test, yhat_svm)


#DECISION TREE CLASSIFIER
paramaters3={'criterion': ['gini', 'entropy'],#how model ensures quality of split:gini->gini impirity,entropy->
             #information gain(shannon entropy)
     'splitter': ['best', 'random'], #strategy to choose split at each node, best->best split,random->random split
     'max_depth': [2*n for n in range(1,10)],#max depth of the tree, generates [2,4,6,8,10,12,14,16,18]..helps prevent overfitting.
     'max_features': ['auto', 'sqrt'], #max number of features consideered for splitting, auto->use all features,sqrt->sqrt(total features) 
     'min_samples_leaf': [1, 2, 4], #minimum samples required at leaf node.[1,2,4] ensures leaves arent too small(prevents overfitting)
     'min_samples_split': [2, 5, 10]} #minimum samples required to split a node.[2,5,10] splitting nodes with too few samples.
tree=DecisionTreeClassifier()
tree_cv=GridSearchCV(tree,param_grid=paramaters3,cv=10)
tree_cv.fit(X_train,Y_train)
print("Best parameters:",tree_cv.best_params_)
print("Accuracy:",tree_cv.best_score_)
best_tree=tree_cv.best_estimator_
accuracy_test_tree=best_tree.score(X_test,Y_test)
print("Test set accuracy of best model:",accuracy_test_tree)
yhat_tree=best_tree.predict(X_test)
print("Predicted values:",yhat_tree)
print("True values:",Y_test)
print("Training set score:", best_tree.score(X_train, Y_train))
print("Test set score:", best_tree.score(X_test, Y_test))
print("Mean Squared Error:", mean_squared_error(Y_test, yhat_tree))
print("R2 Score:", r2_score(Y_test, yhat_tree))
plot_confusion_matrix(Y_test, yhat_tree)



paramaters4={'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],#number of neighbors considered for voting.knn predicts the label by majority 
             #vote among nearest neighbors.trying 10 neighbors.
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
              #how distances between points are calculated, auto->automatically selects best algorithm
              #ball_tree->efficient for high dimensional data,#kd_tree->efficient for low dimensional data
              #brute->brute force distance calcualtion.
              'p': [1,2]} #p means , the distance metric used, p=1->manhattan distance(sum of absolute differences),
#p=2->euclidian distance(straight line distance)

KNN=KNeighborsClassifier()
KNN_cv=GridSearchCV(KNN,param_grid=paramaters4,cv=10)
KNN_cv.fit(X_train,Y_train)
print("Best parameters:",KNN_cv.best_params_)
print("Accuracy:",KNN_cv.best_score_)
best_KNN=KNN_cv.best_estimator_
accuracy_test_KNN=best_KNN.score(X_test,Y_test)
print("Test set accuracy of best model:",accuracy_test_KNN)
yhat_KNN=best_KNN.predict(X_test)
print("Predicted values:",yhat_KNN)
print("True values:",Y_test)
print("Training set score:", best_KNN.score(X_train, Y_train))
print("Test set score:", best_KNN.score(X_test, Y_test))
print("Mean Squared Error:", mean_squared_error(Y_test, yhat_KNN))
print("R2 Score:", r2_score(Y_test, yhat_KNN))
plot_confusion_matrix(Y_test, yhat_KNN)


results = {
    'Model': ['Logistic Regression', 'SVC', 'Decision Tree', 'KNN'],
    'Test Accuracy': [
        best_lr.score(X_test,Y_test),
        best_svm.score(X_test,Y_test),
        best_tree.score(X_test,Y_test),
        best_KNN.score(X_test,Y_test)
    ],
    'Training Accuracy': [
        best_lr.score(X_train,Y_train),
        best_svm.score(X_train,Y_train),
        best_tree.score(X_train,Y_train),
        best_KNN.score(X_train,Y_train)
    ]
}

results_df = pd.DataFrame(results)
print(results_df)
