import scipy.io as io
from sklearn.metrics import mean_absolute_error
from pylab import mpl                              # 正常显示中文
mpl.rcParams['font.sans-serif'] = ['SimHei']   # 正常显示符号
from matplotlib import rcParams
rcParams['axes.unicode_minus']=False
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn import tree
from sklearn import svm
from sklearn import neighbors
from sklearn import ensemble
import xgboost as xgb                                                                   #Xgboost Regressor
model_DecisionTreeRegressor = tree.DecisionTreeRegressor()                               #Decision Tree Regressor
model_SVR = svm.SVR(gamma='auto')                                                                    #SVM Regressor
model_KNeighborsRegressor = neighbors.KNeighborsRegressor()                              #K Neighbors Regressor
model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=20)            #Random Forest Regressor
model_AdaBoostRegressor = ensemble.AdaBoostRegressor(n_estimators=50)                    #Adaboost Regressor
model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(n_estimators=100)   #Gradient Boosting Random Forest Regressor
model_BaggingRegressor = ensemble.BaggingRegressor()                                     #Bagging Regressor
model_ExtraTreeRegressor = tree.ExtraTreeRegressor()                                     #ExtraTree Regressor


def linear_model(X_train,y_train):
    regr =LinearRegression()
    regr.fit(X_train,y_train)
    y_pred = regr.predict(X_train)
    y_test = y_train
    print("linear score on training set: ", mean_absolute_error(y_test, y_pred))
    #plt.figure(figsize=(14,4))
    #plt.scatter(X_train, y_train, color='g')
    #plt.plot(X_train, y_pred, color='r')
    #plt.xlabel('time（0-24）')
    #plt.ylabel('blood glucose value')
    #plt.show()
def polynomial(X_train,y_train):
    pf = PolynomialFeatures(degree=10)
    regr2 = LinearRegression()
    regr2.fit(pf.fit_transform(X_train), y_train)
    #X_predict = np.linspace(0, 24, 1440)
    #y_pred = regr2.predict(pf.transform(X_predict.reshape(X_predict.shape[0], 1)))
    y_pred = regr2.predict(X_train)
    y_test = y_train
    print("polynomial linear score on training set: ", mean_absolute_error(y_test, y_pred))
def lasso(X_train,y_train):
    lasso = Lasso(alpha=.0001, normalize=True, max_iter=1e7)
    lasso = lasso.fit(X_train, y_train)

    y_pred = lasso.predict(X_train)
    y_test = y_train
    print("Lasso score on training set: ", mean_absolute_error(y_test, y_pred))
def decisionTree(X_train,y_train):
    regr =model_DecisionTreeRegressor
    regr.fit(X_train,y_train)
    y_pred = regr.predict(X_train)
    y_test = y_train
    print("decisionTree score on training set: ", mean_absolute_error(y_test, y_pred))
def SVR(X_train,y_train):
    regr =model_SVR
    regr.fit(X_train,y_train)
    y_pred = regr.predict(X_train)
    y_test = y_train
    print("SVR score on training set: ", mean_absolute_error(y_test, y_pred))
def KNeighbors(X_train,y_train):
    regr =model_KNeighborsRegressor
    regr.fit(X_train,y_train)
    y_pred = regr.predict(X_train)
    y_test = y_train
    print("KNeighbors score on training set: ", mean_absolute_error(y_test, y_pred))
def RandomForestRegressor(X_train,y_train):
    regr =model_RandomForestRegressor
    regr.fit(X_train,y_train)
    y_pred = regr.predict(X_train)
    y_test = y_train
    print("RandomForestRegressor score on training set: ", mean_absolute_error(y_test, y_pred))
def AdaBoostRegressor(X_train,y_train):
    regr =model_AdaBoostRegressor
    regr.fit(X_train,y_train)
    y_pred = regr.predict(X_train)
    y_test = y_train
    print("AdaBoostRegressor score on training set: ", mean_absolute_error(y_test, y_pred))
def GradientBoostingRegressor(X_train,y_train):
    regr =model_GradientBoostingRegressor
    regr.fit(X_train,y_train)
    y_pred = regr.predict(X_train)
    y_test = y_train
    print("GradientBoostingRegressor score on training set: ", mean_absolute_error(y_test, y_pred))
def BaggingRegressor(X_train,y_train):
    regr =model_BaggingRegressor
    regr.fit(X_train,y_train)
    y_pred = regr.predict(X_train)
    y_test = y_train
    print("BaggingRegressor score on training set: ", mean_absolute_error(y_test, y_pred))
def ExtraTreeRegressor(X_train,y_train):
    regr =model_ExtraTreeRegressor
    regr.fit(X_train,y_train)
    y_pred = regr.predict(X_train)
    y_test = y_train
    print("ExtraTreeRegressor score on training set: ", mean_absolute_error(y_test, y_pred))
def xgboost(X_train,y_train):
    regr = xgb.XGBRegressor()
    regr.fit(X_train, y_train)

    # Run prediction on training set to get a rough idea of how well it does.
    y_pred = regr.predict(X_train)
    y_test = y_train
    print("XGBoost_in score on training set: ", mean_absolute_error(y_test, y_pred))

def main():
    train1 = io.loadmat(r'D:\tianchiAI\Metro_count\train4\x.mat')['x']
    y_in = io.loadmat(r'D:\tianchiAI\Metro_count\train4\y_in.mat')['y_in']
    scaler = MinMaxScaler()
    train = scaler.fit_transform(train1)
    #test_minmax = scaler.transform(test)
    #y_out = io.loadmat(r'D:\tianchiAI\Metro_count\train4\y_out.mat')['y_out']
    #test = io.loadmat(r'D:\tianchiAI\Metro_count\train4\x_test1')['x_test']
    linear_model(train,y_in)
    lasso(train,y_in)
    decisionTree(train,y_in)
    #SVR(train,y_in)
    KNeighbors(train,y_in)
    RandomForestRegressor(train,y_in)
    AdaBoostRegressor(train,y_in)
    GradientBoostingRegressor(train,y_in)
    BaggingRegressor(train,y_in)
    ExtraTreeRegressor(train,y_in)
    xgboost(train,y_in)

if __name__=='__main__':
    #main()
    train1 = io.loadmat(r'D:\tianchiAI\Metro_count\train4\x.mat')['x']
    y_in = io.loadmat(r'D:\tianchiAI\Metro_count\train4\y_in.mat')['y_in']
    scaler = MinMaxScaler()
    train = scaler.fit_transform(train1)

    rfr_best = model_RandomForestRegressor
    params = {'n_estimators': range(10, 20, 1)}
    gs = GridSearchCV(rfr_best, params, cv=4)
    gs.fit(train1, y_in)
