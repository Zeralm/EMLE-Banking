__version__ = "3.6.5"

import pandas as pd 
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import datetime
import nltk
import matplotlib.pyplot as plt
data = pd.read_csv("Assignments/CodesForCourse.csv")

print("DATA OVERVIEW:")

data["ID"] = range(len(data))
print(data)

print(f"There are {len(data['issuer_eng'].unique())} distinct entries for issuer names")


#QUESTION 1 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+--+-+--+-+-+-+-+-+-+

#Average length issuer field
print("1 ------------------------------------")
avg = pd.Series([len(k) for k in data.loc[:,"issuer_eng"]]).mean()
#Average typing speed per minute: 40 for an average person
minutes_saved_year = (avg/80)*100000
cost_year = minutes_saved_year * 0.05
print(f"{round(cost_year,2)}€ saved a year")


#QUESTION 2 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
print("2 ------------------------------------")
x,X,y,Y = train_test_split(data[["ID","dept_code","issue_date"]],data[["ID","issuer_eng"]],test_size=0.3)

print(f" size Train set = {len(x)}", f" size Test set = {len(X)}")



def crappy_predict(x,X,y,Y):
    #Get a code X, checks its dates. Finds the observation closest date in x for the same code. Gets the issuer_eng of said observation.

    #Gets all observations corresponding to every code of test set
    codesX = X["dept_code"].unique()
    codesx = x["dept_code"].unique()    #Gets all codes

    exceptions = [l for l in codesX if l not in codesx] #Stores the X codes that aren't present in x
    if len(exceptions) == 0: print("Every code present in Test set is present in Train set as well")
    else: print(f"There are {len(exceptions)} codes present in Test set that are not present in Test set")
    #Code: 20003

    #Gets all observations in x and X with that code. 
    utsang = x.loc[[akak == 20003 for akak in x.loc[:,"dept_code"]],:]
    a = pd.DataFrame({"issuer_eng":y.loc[[akak == 20003 for akak in x.loc[:,"dept_code"]],"issuer_eng"]})
    utsang = utsang.join(a["issuer_eng"])
    
    karadel = X.loc[[akak == 20003 for akak in X.loc[:,"dept_code"]],:]

    Xdates = [datetime.datetime.strptime(h, '%Y-%m-%d') for h in karadel.loc[:,"issue_date"]]
    xdates = [datetime.datetime.strptime(g, '%Y-%m-%d') for g in utsang.loc[:,"issue_date"]]

    rightx = pd.Series([datetime.date.isoformat(hue - min([hue-yay for yay in xdates],key=abs)) for hue in Xdates])

    predictions =[]
    for i in range(len(rightx)): predictions.append(list(utsang.loc[[rightx[i] == u for u in utsang["issue_date"]],"issuer_eng"])[0])
    predictions = pd.Series(predictions)
    
    levenstein = [nltk.edit_distance(Y.loc[Y["ID"].isin(karadel["ID"]),"issuer_eng"].iloc[j],predictions.iloc[j]) for j in range(len(predictions))]
    
    performance = [[u<=o for u in levenstein].count(True)/len(levenstein) for o in range(50)]
    
    plt.plot(range(50),performance)
    plt.show()  
    
    
crappy_predict(x,X,y,Y)
