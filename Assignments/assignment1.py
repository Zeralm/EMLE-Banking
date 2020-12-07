import pandas as pd 
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import datetime


data = pd.read_csv("C:/Users/David/Desktop/Machine/CodesForCourse.csv")



data["ID"] = range(len(data))
print(data)

print(len(data[["issuer_eng","issue_date"]]))

print(len(data["issuer_eng"].unique()))


#QUESTION 1 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+--+-+--+-+-+-+-+-+-+

#Average length issuer field
avg = pd.Series([len(k) for k in data.loc[:,"issuer_eng"]]).mean()
#Average typing speed per minute: 40 for an average person
print(avg)
minutes_year = (avg/80)*100000
cost_year = minutes_year * 0.05
print(f"{cost_year}€ a year")


#QUESTION 2 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

x,X,y,Y = train_test_split(data[["ID","dept_code","issue_date"]],data[["ID","issuer_eng"]],test_size=0.3)

print((f" length x = {len(x)}", f" length X = {len(X)}"))



def crappy_predict(x,X,y,Y):
    #Get a code X, checks its dates. Finds the observation closest date in x for the same code. Gets the issuer_eng of said observation.

    #Gets all observations corresponding to every code of test set
    codesX = X["dept_code"].unique()
    codesx = x["dept_code"].unique()    #Gets all codes

    exceptions = [l for l in codesX if l not in codesx] #Stores the X codes that aren't present in x

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
    
    accuracy = [Y.loc[Y["ID"].isin(karadel["ID"]),"issuer_eng"].iloc[j] == predictions.iloc[j] for j in range(len(predictions))].count(True)/len(predictions)
    print(f"{accuracy} accuracy")
    #Gets the issue_eng of the observations with the minimum difference between the dates of the dates corresponding to those observations and the corresponding dates in X
    #Assigns to Y each issue_eng found for the corresponding X
                

    #Gets all observations corresponding to every code of train set
    
crappy_predict(x,X,y,Y)
