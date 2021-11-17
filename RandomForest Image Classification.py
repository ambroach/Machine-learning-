# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 19:19:22 2021

@author: abbas
"""
import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from tkinter import *
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report
import seaborn as sns
from collections import Counter

#'BUBBLE MINT GUM', 'CHEESE' ,'BASMATI RICE', 'CHOCO BALLS CEREALS', 'CHOCOLATE ECLAIR', 'FRIED EGG', 'GARLIC SAUCE'
Categories=['APPLE CAKE', 'ASSORTED SALAD', 'BAKED POTATO', 'BAKED TROUT FISH', 'BARBEQUE DRUMSTICKS', 'BARBEQUE SAUCE', 'BARBEQUE WINGS', 'BASMATI RICE', 'BUBBLE MINT GUM', 'CHEESE', 'CHEESE BALLS', 'CHICKEN FAJITAS', 'CHILLI SAUCE', 'CHOCO BALLS CEREALS', 'CHOCOLATE ECLAIR', 'COOKED COUSCOUS WITH DILL', 'CRANBERRY COOKIES WITH WHITE CHOCOLATE', 'CRISPY CHICKEN', 'CRUNCHY TACO', 'DONUTS WITH VANILLA SUGAR POWDER', 'FISH FINGERS', 'FRENCH FRIES', 'FRIED EGG', 'FRIED POTATO WEDGES', 'GARLIC SAUCE', 'GRILLED CHICKEN', 'HOT SALAMI SANDWICH WITH KETCHUP AND GARLIC SAUCES', 'KFC CRISPY', 'KFC REAL BURGER', 'KITKAT CHOCOLATE', 'LOLLIPOPS', 'MASHED POTATOES', 'NAKED CHICKEN TACO', 'PAKORA', 'POPCORN WITH CARAMEL', 'PORK SAUSAGES', 'PRETZEL WITH SALT', 'RAFFAELLO BALLS', 'STEAMED FISH (PASTRAV)', 'STRAWBERRY CHEESECAKE', 'STRAWBERRY MOUSSE CAKE', 'TURKEY HAM', 'VEGETABLE CURRY', 'WHIPPED CREAM ICE CREAM', 'WHOLE WHEAT OAT MUFFINS', 'WHOLEGRAIN BAGUETTE WITH FLAX SEEDS', 'YELLOW MUSTARD', 'ZUCCHINI SOUP', 'BIG CROISSANT WITH STRAWBERRY FILLING', 'ROASTED PIG']
flat_data_arr=[] #input array
target_arr=[] #output array
datadir=r'C:\Users\abbas\Desktop\images- ML - Copy'


for i in Categories:
    path = os.path.join(datadir,i)
    for img in os.listdir(path):
           img_array = imread(os.path.join(path,img))
           img_resized=resize(img_array, (100,100,3))
           flat_data_arr.append(img_resized.flatten())
           target_arr.append(Categories.index(i))
     
totalFiles = 0
totalDir = 0

list2 = []

c = Counter(target_arr) 

for i in range(0,50):
    list2.append(c[i])



fig = plt.figure(figsize = (10, 10)) 
  
# creating the bar plot 
plt.bar(Categories, list2, color ='green',  
        width = 0.8) 
plt.xticks(rotation=90)
plt.xlabel("Different categories in the dataset") 
plt.ylabel("Number of Images in each class") 
plt.title("Images and Categories") 
plt.show() 

for base, dirs, files in os.walk(datadir):
    for directories in dirs:
        totalDir += 1
    for Files in files:
        totalFiles += 1

print('Total number of Images',totalFiles)
print('Total Number of directories',totalDir)
   
    
    

df=pd.DataFrame(flat_data_arr) #dataframe
df['Target']=target_arr

x = df.iloc[:, 0:3].values
y = df.iloc[:, 2].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=0)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)

y_predicts = regressor.predict(x_test)
print(y_predicts)


accuracy = regressor.score(x_test, y_test)
print( 'Random Forest Algorithm: ', accuracy*100,'%')

#print(confusion_matrix(regressor, y_test,y_pred))

Display = Tk()
Display.title('Image Classification')
Display.geometry("600x600")
Display.configure(bg='white')
Display.resizable(True,True)

label = Label(Display, bg=None, font=("Courier", 20, "bold"), relief="flat")
label.place(x=10,y=10)
label.configure(text="Image Classification")
 
Text_for_display = 'Accuracy: ' + str(accuracy*100)
 
Score = Label(Display, bg="white", font=("Courier", 14), relief="flat")
Score.place(x=100,y=100)
Score.configure(text=Text_for_display)

Display.mainloop()
