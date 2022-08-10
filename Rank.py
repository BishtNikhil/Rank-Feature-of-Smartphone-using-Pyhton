import inline as inline
import matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import plotly.express as px

data = pd.read_csv(r'D:\NIKHIL\Amity University Noida\Rank Feature\MobileTrain.csv')
data.head()

# getting information of the dataset like non-null count and Data type of each column
data.info()

# Summary of the data
data.describe()

# checking number null values in the dataset
data.isna().sum()

# getting number of unique values in the dataset
data.nunique()

plt.figure()
sns.barplot(y="battery_power", x="price_range", data=data)

for i in data:
    if (data[i].nunique())<=30:
        sns.countplot(x=data[i])
        plt.show()

plt.figure(figsize=(30,10))
plt.subplot(331)
sns.distplot(data["battery_power"])

plt.subplot(332)
sns.distplot(data["clock_speed"])

plt.subplot(333)
sns.distplot(data["int_memory"])

plt.subplot(334)
sns.distplot(data["mobile_wt"])

plt.subplot(335)
sns.distplot(data["px_height"])

plt.subplot(336)
sns.distplot(data["px_width"])

plt.subplot(337)
sns.distplot(data["ram"])

plt.subplot(338)
sns.distplot(data["talk_time"])

plt.subplot(339)
sns.distplot(data["m_dep"])

plt.show()

dataset = data
dataset.head()

dataset["isBluetooth"]=''
for i in range(len(dataset)):
    if dataset["blue"][i]==0:
        dataset["isBluetooth"][i]="No"
    else:
        dataset["isBluetooth"][i]="Yes"
px.pie(data_frame=dataset, names="isBluetooth", title="Percentage of devices having bluetooth", hole=0.3)

dataset["isDualSIM"]=''
for i in range(len(dataset)):
    if dataset["dual_sim"][i]==0:
        dataset["isDualSIM"][i]="No"
    else:
        dataset["isDualSIM"][i]="Yes"
px.pie(data_frame=dataset, names="isDualSIM", title="Percentage of devices having Dual SIM", hole=0.3)

dataset["is_4G"]=''
for i in range(len(dataset)):
    if dataset["four_g"][i]==0:
        dataset["is_4G"][i]="No"
    else:
        dataset["is_4G"][i]="Yes"
px.pie(data_frame=dataset, names="is_4G", title="Percentage of devices having 4G", hole=0.3)

dataset["isWiFi"]=''
for i in range(len(dataset)):
    if dataset["wifi"][i]==0:
        dataset["isWiFi"][i]="No"
    else:
        dataset["isWiFi"][i]="Yes"
px.pie(data_frame=dataset, names="isWiFi", title="Percentage of devices having Dual SIM", hole=0.3)

dataset["Cores"]=''
for i in range(len(dataset)):
    if dataset["n_cores"][i]==1:
        dataset["Cores"][i]="Single Core"
    elif dataset["n_cores"][i]==2:
        dataset["Cores"][i]="Dual Core"
    elif dataset["n_cores"][i]==3:
        dataset["Cores"][i]="Triple Core"
    elif dataset["n_cores"][i]==4:
        dataset["Cores"][i]="Quad Core"
    elif dataset["n_cores"][i]==5:
        dataset["Cores"][i]="Penta Core"
    elif dataset["n_cores"][i]==6:
        dataset["Cores"][i]="Hexa Core"
    elif dataset["n_cores"][i]==7:
        dataset["Cores"][i]="Hepta Core"
    else:
        dataset["Cores"][i]="Octa Core"
px.pie(data_frame=dataset, names="Cores", title="Percentage of devices having Dual SIM", hole=0.3)

#Histogram using plotly.express library on various columns
px.histogram(data_frame=dataset, x="isBluetooth", color="price_range", title="Comparison according to price range of bluetooth devices or not")

px.histogram(data_frame=dataset, x="isDualSIM", color="price_range", title="Comparison according to price range of Dual SIM devices or not")

px.histogram(data_frame=dataset, x="is_4G", color="price_range", title="Comparison according to price range of 4G devices or not")

px.histogram(data_frame=dataset, x="Cores", color="price_range", title="Comparison according to price range of number of cores per device")

px.histogram(data_frame=dataset, x="isWiFi", color="price_range", title="Comparison according to price range of WiFi devices or not")

data1 = data.loc[:,['battery_power','blue','dual_sim','fc','four_g','int_memory','m_dep','mobile_wt','n_cores','pc','px_height','px_width','ram','sc_h','sc_w','price_range']]
data1

data2 = data.loc[:,['talk_time','three_g','wifi','touch_screen','clock_speed']]
data2

#Merging Datasets
data3 = pd.concat([data1, data2], axis=1)
data3

#Ranking dataset according to price range
dt = data
dt["rank_by_price"] = dt["price_range"].rank()
dt1 = dt
dt1.head()

#Sorting above dataset according to ranked_price_range
dt1.sort_values(by=["rank_by_price"])

#Ranking on all the features using rank()
dt2 = data
RankedDataset1 = dt2.rank()
RankedDataset1.sort_values(by="price_range")

#Ranking all the features separately to correct output
# because not all features are good when values are high and not all features are good when values are low
# It depends on each and every feature

b = dt2
b["rank_by_price"] = b["price_range"].rank()
b["rank_by_battery"] = b["battery_power"].rank(ascending=False)
b["rank_by_blueooth"] = b["blue"].rank(ascending=False)
b["rank_by_clockspeed"] = b["clock_speed"].rank(ascending=False)
b["rank_by_DualSIM"] = b["dual_sim"].rank(ascending=False)
b["rank_by_fc"] = b["fc"].rank(ascending=False)
b["rank_by_4G"] = b["four_g"].rank(ascending=False)
b["rank_by_InternalMemory"] = b["int_memory"].rank(ascending=False)
b["rank_by_mdep"] = b["m_dep"].rank(ascending=False)
b["rank_by_weight"] = b["mobile_wt"].rank(ascending=True)
b["rank_by_ncores"] = b["n_cores"].rank(ascending=False)
b["rank_by_pc"] = b["pc"].rank(ascending=False)
b["rank_by_height"] = b["px_height"].rank(ascending=False)
b["rank_by_width"] = b["px_width"].rank(ascending=False)
b["rank_by_ram"] = b["ram"].rank(ascending=False)
b["rank_by_sch"] = b["sc_h"].rank(ascending=False)
b["rank_by_scw"] = b["sc_w"].rank(ascending=False)
b["rank_by_talktime"] = b["talk_time"].rank(ascending=False)
b["rank_by_3G"] = b["three_g"].rank(ascending=False)
b["rank_by_touchscreen"] = b["touch_screen"].rank(ascending=False)
b["rank_by_wifi"] = b["wifi"].rank(ascending=False)
b.head()

RankedDataset2 = b.iloc[:,21:]
RankedDataset2

#Data Splitting and Data Scaling
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data=data.iloc[:,:21]

X=data.iloc[:,:-1]
y=data.iloc[:,-1]

sc = StandardScaler()
X=sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

lg = LogisticRegression()
lg.fit(X_train, y_train)
y_pred_lg = lg.predict(X_test)
#print(y_pred_lg)
print("Trainig Score of Logistic Regression is {}".format(lg.score(X_train, y_train)*100))
as_lg = accuracy_score(y_test, y_pred_lg)*100
print("Accuracy of Logistic Regression classifier is {}".format(as_lg))
print("Confusion matrix of Logistic Regression classifier : \n{}".format(confusion_matrix(y_test, y_pred_lg)))
print("\n")
print("{}".format(classification_report(y_test, y_pred_lg)))

#Naive Bayes
from sklearn.naive_bayes import GaussianNB
Nb = GaussianNB()
Nb.fit(X_train, y_train)
print("Traning Score of Naive Bayes is {}".format(Nb.score(X_train, y_train)*100))
y_pred_Nb = Nb.predict(X_test)
print(y_pred_Nb)
as_Nb = accuracy_score(y_test, y_pred_Nb)*100
print("Accuracy of Naive Bayes classifier is {}".format(as_Nb))
print("Confusion matrix of Naive Bayes : \n{}".format(confusion_matrix(y_test, y_pred_Nb)))
print("\n")
print("{}".format(classification_report(y_test, y_pred_Nb)))

#SVM
from sklearn.svm import SVC
svc = SVC(kernel="rbf")
svc.fit(X_train, y_train)
print("Traning Score of SVM is {}".format(svc.score(X_train, y_train)*100))
y_pred_svm = svc.predict(X_test)
print(y_pred_svm)
am_svm = accuracy_score(y_test, y_pred_svm)*100
print("Accuracy of SVM classifier is {}".format(am_svm))
print("Confusion matrix of SVM classifier : \n{}".format(confusion_matrix(y_test, y_pred_svm)))
print("\n")
print("{}".format(classification_report(y_test, y_pred_svm)))

#Desision Tree
from sklearn.tree import DecisionTreeClassifier
Dt = DecisionTreeClassifier(criterion="entropy")
Dt.fit(X_train, y_train)
print("Traning Score of Decision Tree classifier is {}".format(Dt.score(X_train, y_train)*100))
y_pred_Dt = Dt.predict(X_test)
print(y_pred_Dt)
am_Dt = accuracy_score(y_test, y_pred_Dt)*100
print("Accuracy of Decision Tree classifier is {}".format(am_Dt))
print("Confusion matrix of Dcision Tree classifier : \n{}".format(confusion_matrix(y_test, y_pred_Dt)))
print("\n")
print("{}".format(classification_report(y_test, y_pred_Dt)))

#Random Forest
from sklearn.ensemble import RandomForestClassifier
Rf = RandomForestClassifier(n_estimators=300)
Rf.fit(X_train, y_train)
print("Traning Score of Random Forest classifier is {}".format(Rf.score(X_train, y_train)*100))
y_pred_Rf = Rf.predict(X_test)
print(y_pred_Rf)
am_Rf = accuracy_score(y_test, y_pred_Rf)*100
print("Accuracy of Decision Tree classifier is {}".format(am_Rf))
print("Confusion matrix of Dcision Tree classifier : \n{}".format(confusion_matrix(y_test, y_pred_Rf)))
print("\n")
print("{}".format(classification_report(y_test, y_pred_Rf)))

#Comparing Accuracies of all the models using Bar Chart
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

classifiers = ["Logistic Regression", "Naive Bayes", "SVM", "Decision Tree", "Random Forest"]
Acc_list = [as_lg, as_Nb, am_svm, am_Dt, am_Rf]
df_models = pd.DataFrame({"Model":classifiers, "Accuracy":Acc_list})
px.histogram(data_frame=df_models, x="Model", y="Accuracy", color=["red","yellow","blue","orange","green"])

#Performing K-Fold and Cross Validation on all the algorithms to get better accuracy and result
K_Fold = KFold(n_splits=10)
abc=[]
Acc = []
Classifiers = ["Logistic Regression", "Naive Bayes", "SVM", "Decision Tree", "Random Forest"]
Models = [LogisticRegression(), GaussianNB(), SVC(kernel="rbf"), DecisionTreeClassifier(criterion="entropy"), RandomForestClassifier(n_estimators=300)]
for i in Models:
    model = i
    CV_Result = cross_val_score(model, X_train, y_train, cv=K_Fold, scoring="accuracy")
    abc.append(CV_Result.mean())
    Acc.append(CV_Result)

CV_ModelData = pd.DataFrame(abc, index=Classifiers)
CV_ModelData.columns = ["CV Mean"]
CV_ModelData

#Plotting Box Plot Accuracies we got from Cross Validation and K-fold
box = pd.DataFrame(Acc, index=[Classifiers])
boxT = box.T
plt.figure(figsize=(10,8))
ax = sns.boxplot(data=boxT, orient="v", palette="Set2", width=.6)