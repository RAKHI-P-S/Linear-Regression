import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model


rg=pd.read_csv('Dataset/homeprices.csv')

reg=linear_model.LinearRegression()
reg.fit(rg[["area"]],rg.price)
predicted_price=reg.predict([[5000]])
print("The predictedprice is",predicted_price[0])

plt.xlabel("Area")
plt.ylabel("Price")
scter=plt.scatter(rg.area,rg.price,color='red',marker="*")
plots=plt.plot(rg.area,reg.predict(rg[["area"]]),color='blue')
plt.show()

cof=reg.coef_[0]
print("Slope (m):",cof)
intr=reg.intercept_
print("Intercept (c):",intr)
# y=m*X+b
y=cof*5000+intr
print(y)
print("✅ Code executed successfully!")

d=pd.read_csv('Dataset/areas.csv')
df=reg.predict(d)
print(df)
d['price']=df
print(d)
d.to_csv("predict.csv")