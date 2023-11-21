# -*- coding: utf-8 -*-
"""
@author: Hao WAN
eneral overview
Pred_MMM (CO2) is an interactive desktop application for predicting the associated separation properties of CO2 gas molecules in MMM materials. The core of its calculation is the Stacking model that we trained. Below is the source code for the interface design.
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tkinter as tk 
import ttkbootstrap as ttk
from tkinter import filedialog
from PIL import Image,ImageTk
import webbrowser
import joblib
import numpy as np
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

## Load the trained Stacking model
meta_model0 = joblib.load("model/meta_model0.pt")## The detailed code for model training is in “Stacking_code.py”
model_01 = joblib.load("model/model_01.pt")
model_02 = joblib.load("model/model_02.pt")
meta_model1 = joblib.load("model/meta_model1.pt")
model_11 = joblib.load("model/model_11.pt")
model_12 = joblib.load("model/model_12.pt")
meta_model2 = joblib.load("model/meta_model2.pt")
model_21 = joblib.load("model/model_21.pt")
model_22 = joblib.load("model/model_22.pt")
meta_model3 = joblib.load("model/meta_model3.pt")
model_31 = joblib.load("model/model_31.pt")
model_32 = joblib.load("model/model_32.pt")
meta_model4 = joblib.load("model/meta_model4.pt")
model_41 = joblib.load("model/model_41.pt")
model_42 = joblib.load("model/model_42.pt")
scaler = joblib.load("model/scaler.pkl")

## The layout design of the main interface
root= tk.Tk()
root.title("Predict properties of MMM material on stacking model")
root.resizable(True,True) ## The window size can be changed
canvas1 = tk.Canvas(root, width = 820, height =730) ## Main window size
canvas1.pack()
## The plate layout within the main interface
re1=canvas1.create_rectangle(50,20,770,450)
re2=canvas1.create_rectangle(50,470,770,670)
re3=canvas1.create_rectangle(70,100,480,320,outline='darkgray')
re4=canvas1.create_rectangle(500,100,755,320,outline='darkgray')
re4=canvas1.create_rectangle(70,340,755,430,outline='darkgray')
re4=canvas1.create_rectangle(70,560,755,650,outline='darkgray')
label_B = tk.Label(root,font=('microsoft yahei',10),text='Predicted results')
canvas1.create_window(620, 340, window=label_B)
label_B = tk.Label(root,font=('microsoft yahei',10),text='Predicted results')
canvas1.create_window(620, 560, window=label_B)
label_B = tk.Label(root,font=('microsoft yahei',9),text='Author：Hao WAN,Zhiwei Qiao,Guangzhou University')
canvas1.create_window(620, 715, window=label_B)

scaled_data = None
Prediction_result = None
def update_prediction():
    global scaled_data, Prediction_result
    selected_option = var1.get()  # Gets the selected options
    if selected_option:
        #Perform the corresponding prediction operations based on the selected model
        if selected_option == "TSP(CO2/N2)":
            test_predictions_1 = model_11.predict(scaled_data)
            test_predictions_2 = model_12.predict(scaled_data)
            test_feature_matrix = np.column_stack((test_predictions_1, test_predictions_2))
            R = meta_model1.predict(test_feature_matrix)
        elif selected_option == "TSP(CO2/CH4)":
            test_predictions_1 = model_21.predict(scaled_data)
            test_predictions_2 = model_22.predict(scaled_data)
            test_feature_matrix = np.column_stack((test_predictions_1, test_predictions_2))
            R = meta_model2.predict(test_feature_matrix)
        elif selected_option == "TSP(CO2/O2)":
            test_predictions_1 = model_31.predict(scaled_data)
            test_predictions_2 = model_32.predict(scaled_data)
            test_feature_matrix = np.column_stack((test_predictions_1, test_predictions_2))
            R = meta_model3.predict(test_feature_matrix)
        elif selected_option == "TSP(CO2/H2)":
            test_predictions_1 = model_41.predict(scaled_data)
            test_predictions_2 = model_42.predict(scaled_data)
            test_feature_matrix = np.column_stack((test_predictions_1, test_predictions_2))
            R = meta_model4.predict(test_feature_matrix)
        elif selected_option == "P":
            test_predictions_1 = model_01.predict(scaled_data)
            test_predictions_2 = model_02.predict(scaled_data)
            test_feature_matrix = np.column_stack((test_predictions_1, test_predictions_2))
            R = meta_model0.predict(test_feature_matrix)
    else:
        # The user does not select any options and does not make predictions
        R = ""

    if R is None or R == "":
        R3 = "No prediction available"
    else:
        R1 = float(R)
        R2 = format(R1, '.2E')
        R3 = R2.split('E')
        if R3[1][0] == "-":
            R3 = R3[0] + " × 10^" + R3[1].lstrip('0')
        else:
            R3 = R3[0] + " × 10^" + R3[1][1:].lstrip('0')
        
        #Update the text of the prediction result label
        Prediction_result.set(R3)

def values():
    global scaled_data, Prediction_result 
    Prediction_result = tk.StringVar()
    # Gets the value of the input box and converts it to a floating point number
    New_HVF = float(entry1.get()) 
    New_PLD = float(entry2.get()) 
    New_LCD = float(entry3.get()) 
    New_Density = float(entry4.get()) 
    New_PSD = float(entry5.get()) 
    New_LCDPLD = float(entry6.get()) 
    New_Qst = float(entry7.get()) 
    New_VSA = float(entry8.get())
    New_FFV = float(entry9.get())
    New_polyDES = float(entry10.get())
    New_MOFpoly = float(entry11.get())

    # Create a new prediction result label
    Prediction_result = tk.StringVar()  # Create a StringVar variable to bind the text of the prediction result label
    label_Prediction = tk.Label(root, font=('microsoft yahei', 12), width=30, height=2, textvariable=Prediction_result)
    canvas1.create_window(600, 380, window=label_Prediction)
    
    # unit label
    lbo2 = tk.Label(root, font=('microsoft yahei', 12), text='(barrer)')
    canvas1.create_window(700, 380, window=lbo2)

    new_data = [[New_LCD, New_HVF, New_VSA, New_PLD, New_LCDPLD, New_Density, New_PSD, New_Qst, New_MOFpoly, New_polyDES, New_FFV]]
    scaled_data = scaler.transform(new_data)
    
    update_prediction()  
    
## Sets the label and entry for entering the nine descriptor 
label_Z = tk.Label(root,font=('microsoft yahei',15,'bold'),text='Predict the properties of MMM material(CO2)')
canvas1.create_window(415, 20, window=label_Z)

label_L = tk.Label(root,font=('microsoft yahei',10),text='Physical property of MOF material')
canvas1.create_window(290, 100, window=label_L)

label1 = tk.Label(root,font=('microsoft yahei',10),text='HVF:') ## create 1st label box 
canvas1.create_window(110, 140, window=label1)
entry1 = tk.Entry (root,font=('microsoft yahei',10),width=8,justify='center') ## create 1st entry box 
entry1.insert(0, "0.826")
canvas1.create_window(200, 140, window=entry1)

label2 = tk.Label(root,font=('microsoft yahei',10), text='PLD (Å): ') ## create 2st label box 
canvas1.create_window(110, 190, window=label2)
entry2 = tk.Entry (root,font=('microsoft yahei',10),width=8,justify='center') ## create 2nd entry box
entry2.insert(0, "7.83")
canvas1.create_window(200, 190, window=entry2)

label3 = tk.Label(root,font=('microsoft yahei',10), text='LCD (Å): ') ## create 3st label box 
canvas1.create_window(110, 240, window=label3)
entry3 = tk.Entry (root,font=('microsoft yahei',10),width=8,justify='center') ## create 3nd entry box
entry3.insert(0, "14.946")
canvas1.create_window(200, 240, window=entry3)

label4 = tk.Label(root,font=('microsoft yahei',10,"italic"), text='ρ(kg/m^3):') ## create 4st label box 
canvas1.create_window(110, 290, window=label4)
entry4 = tk.Entry (root,font=('microsoft yahei',10),width=8,justify='center') ## create 4nd entry box
entry4.insert(0, "604.652")
canvas1.create_window(200, 290, window=entry4)

label5 = tk.Label(root, font=('microsoft yahei',10),text='PSD%:') ## create 5st label box 
canvas1.create_window(310, 140, window=label5) 
entry5 = tk.Entry (root,font=('microsoft yahei',10),width=8,justify='center') ## create 5nd entry box
entry5.insert(0, "0")
canvas1.create_window(420, 140, window=entry5)

label6 = tk.Label(root, font=('microsoft yahei',10),text='LCD/PLD:') ## create 6st label box 
canvas1.create_window(310, 190, window=label6) 
entry6 = tk.Entry (root,font=('microsoft yahei',10),width=8,justify='center') ## create 6nd entry box
entry6.insert(0, "1.909")
canvas1.create_window(420, 190, window=entry6)

label7 = tk.Label(root, font=('microsoft yahei',10),text='Qst(kJ/mol):') ## create 7st label box 
canvas1.create_window(310, 240, window=label7) 
entry7 = tk.Entry (root,font=('microsoft yahei',10),width=8,justify='center') ## create 7nd entry box
entry7.insert(0, "0.0831")
canvas1.create_window(420, 240, window=entry7)

label8 = tk.Label(root, font=('microsoft yahei',10),text='VSA(m^2/cm^3):') ## create 8st label box 
canvas1.create_window(310, 290, window=label8) 
entry8 = tk.Entry (root,font=('microsoft yahei',10),width=8,justify='center') ## create 8nd entry box
entry8.insert(0, "2130.76")
canvas1.create_window(420, 290, window=entry8)

label_R = tk.Label(root,font=('microsoft yahei',10),text='Physical property of Polymer material') 
canvas1.create_window(630, 100, window=label_R)

label9 = tk.Label(root, font=('microsoft yahei',10,"italic"),text='FFV (cm^3/g):') ## create 9st 6abel box 
canvas1.create_window(570,190, window=label9) 
entry9 = tk.Entry (root,font=('microsoft yahei',11),width=8,justify='center') ## create 9nd entry box
entry9.insert(0, "0.17")
canvas1.create_window(680,190, window=entry9)

label10 = tk.Label(root, font=('microsoft yahei',10,"italic"),text='ρ_poly (kg/m^3):') ## create 10st label box 
canvas1.create_window(570, 240, window=label10)
entry10 = tk.Entry (root,font=('microsoft yahei',11),width=8,justify='center') ## create 10nd entry box
entry10.insert(0, "1240") 
canvas1.create_window(680, 240, window=entry10)

label11 = tk.Label(root, font=('microsoft yahei',10,"italic"),text='ρ (MOF/poly):') ## create 11st label box 
canvas1.create_window(570, 290, window=label11) 
entry11 = tk.Entry (root,font=('microsoft yahei',11),width=8,justify='center') ## create 11nd entry box
entry11.insert(0, "0.4876")
canvas1.create_window(680, 290, window=entry11)

#Create a drop-down box
#var1 = tk.StringVar()  
var1 = tk.StringVar(value="P")
cm1 = ttk.Combobox(root, textvariable=var1, font=('microsoft yahei', 10))
cm1["value"] = ("TSP(CO2/N2)", "TSP(CO2/CH4)", "TSP(CO2/O2)", "TSP(CO2/H2)", "P")
canvas1.create_window(220, 380, window=cm1)

# Bind a drop-down event
cm1.bind('<<ComboboxSelected>>', lambda event: update_prediction())

predict_button = tk.Button(root, text="Predicted",font=('microsoft yahei', 10),command=values)
#predict_button.configure(bg="lightgray",fg="black")
canvas1.create_window(380, 380, window=predict_button)

## Message box (Related literature on molecular physical properties)
def cmx1():
    window = tk.Tk()     
    window.title('Warm prompt')     
    window.geometry('350x250')
    link = tk.Label(window, text='Polymer properties: The FFV(0.132-0.26) and \nρ_poly(1057-1442) ranges was better predicted.\n TSP=S*LN(P)\n The initial values in the input box represent \n(SAHYIK)MOF-5 and Matrimid respectively,\nand the predicted P-values are consistent with\n the experiments.\nYou can change the value of the input field.\nhttps://doi.org/10.1016/j.memsci.2008.12.006.'
             , font=('microsoft yahei',10),anchor="center")
    link.place(x=30, y=50) 
    def open_url(event):
        webbrowser.open("https://doi.org/10.1016/j.memsci.2008.12.006", new=0)         
    link.bind("<Button-1>", open_url)    
btn1=tk.Button(root, text='Tooltip',font=('microsoft yahei',10), command=cmx1)
btn1.configure(bg="red")
canvas1.create_window(630, 140, window=btn1)

##Message box (Instructions for Prediction of a single material diffusiivity)
def resize(w, h, w_box, h_box, pil_image): 
    f1 = 1*w_box/w 
    f2 =1*h_box/h  
    factor = min([f1, f2])  
    width = int(w*factor)  
    height = int(h*factor)  
    return pil_image.resize((width, height),Image.ANTIALIAS)
       
w_box = 600  
h_box = 450    

global tk_image 
photo1 = Image.open("Img/full_name.png")
w, h = photo1.size       
photo1_resized = resize(w, h, w_box, h_box, photo1)    
tk_image1 = ImageTk.PhotoImage(photo1_resized)

def cmx2():
    top2=tk.Toplevel() 
    top2.title('Instructions for use') 
    top2.geometry('620x500') 
    lab_1 = ttk.Label(top2,image=tk_image1) 
    lab_1.place(x=25, y=10) 
    top2.mainloop()  
btn2=tk.Button(root, text='README',font=('microsoft yahei',10), command=cmx2)
btn2.configure(bg="red")
canvas1.create_window(120, 60, window=btn2)

## Message box (Instructions for batch Prediction of material diffusiivity)
def resize(w, h, w_box, h_box, pil_image):
    f1 = 1*w_box/w 
    f2 =1*h_box/h  
    factor = min([f1, f2])  
    width = int(w*factor)  
    height = int(h*factor)  
    return pil_image.resize((width, height), Image.ANTIALIAS)      
w_box = 600  
h_box = 500    

global tk_image 
photo2 = Image.open("Img/sample_file.png")  
w, h = photo2.size     
photo2_resized = resize(w, h, w_box, h_box, photo2)    
tk_image2 = ImageTk.PhotoImage(photo2_resized)

def cmx3():
    top1=tk.Toplevel()
    top1.title('Instructions for use')     
    top1.geometry('680x580')
    lab2 = tk.Label(top1, text='You need to create the data you want to compute\nin the format below (For example：the prediction \nTSPCO2/N2 of MMM:'
                    , font=('microsoft yahei',15),anchor="nw",justify='left')
    lab2.place(x=20, y=20) 
    lab3 = tk.Label(top1, text='After creating the file, you can click the import file \nbutton on the screen.The predicted result  will be \nsaved in "Result/Batch_Predicted.xlsx.\nThe saved prediction table will also be overwritten with\n the new drop-down option".'
                    , font=('microsoft yahei',15),anchor="nw",justify='left')
    lab3.place(x=30, y=400)
    lab_img = tk.Label(top1, image=tk_image2)
    lab_img.place(x=30, y=120)
    #lab3 = ttk.Label(top1,text="photo:",image=tk_image2)
    #lab3.place(x=30, y=120) 
    top1.mainloop()     
btn3=tk.Button(root, text='README',font=('microsoft yahei',10),command=cmx3)
btn3.configure(bg="red")
canvas1.create_window(120, 520, window=btn3)

## Batch prediction of material diffusivity
label_Z1 = tk.Label(root,font=('microsoft yahei',12,'bold'),text='Batch prediction of MMM material properties')
canvas1.create_window(415, 470, window=label_Z1)

## Open File
def open_file():
    filename = filedialog.askopenfilename(title='open exce')
    entry_filename.delete(0,"end")
    entry_filename.insert('insert', filename)

button_import = tk.Button(root, text="Import File",font=('microsoft yahei',10),command=open_file)
canvas1.create_window(320, 520, window=button_import)
 
## Import File
entry_filename = tk.Entry(root,font=('microsoft yahei',10),width=30)
canvas1.create_window(520, 520, window=entry_filename)

def batch_prediction(X_pred, selected_option):
    #global X_pred, Pred_result
    selected_option = var2.get()  
    if selected_option:
        # Perform the corresponding prediction operations based on the selected model
        if selected_option == "TSP(CO2/N2)":
            test_predictions_1 = model_11.predict(X_pred)
            test_predictions_2 = model_12.predict(X_pred)
            test_feature_matrix = np.column_stack((test_predictions_1, test_predictions_2))
            R1 = meta_model1.predict(test_feature_matrix)
        elif selected_option == "TSP(CO2/CH4)":
            test_predictions_1 = model_21.predict(X_pred)
            test_predictions_2 = model_22.predict(X_pred)
            test_feature_matrix = np.column_stack((test_predictions_1, test_predictions_2))
            R1 = meta_model2.predict(test_feature_matrix)
        elif selected_option == "TSP(CO2/O2)":
            test_predictions_1 = model_31.predict(X_pred)
            test_predictions_2 = model_32.predict(X_pred)
            test_feature_matrix = np.column_stack((test_predictions_1, test_predictions_2))
            R1 = meta_model3.predict(test_feature_matrix)
        elif selected_option == "TSP(CO2/H2)":
            test_predictions_1 = model_41.predict(X_pred)
            test_predictions_2 = model_42.predict(X_pred)
            test_feature_matrix = np.column_stack((test_predictions_1, test_predictions_2))
            R1 = meta_model4.predict(test_feature_matrix)
        elif selected_option == "P":
            test_predictions_1 = model_01.predict(X_pred)
            test_predictions_2 = model_02.predict(X_pred)
            test_feature_matrix = np.column_stack((test_predictions_1, test_predictions_2))
            R1 = meta_model0.predict(test_feature_matrix)
    else:
        # The user does not select any options and does not make predictions
         R1 = ""
    return R1

def predict():
    selected_option = var2.get() 
    if selected_option:
        # Get file path
        filename = entry_filename.get()

        # Load Excel file data
        pred_data1 = pd.read_excel(filename)
        pred_data = pred_data1.dropna(axis=0)
        #pred_data[pred_data.isnull()]
        #pred_data[pred_data.notnull()]
        df = pd.DataFrame(pred_data, columns=['LCD', 'HVF', 'VSA', 'PLD', 'LCD/PLD', 'ρ', 'PSD', 'Qst', 'ρ(MOF/poly)', 'ρ_poly', 'FFV'])
        X_new = df[['LCD', 'HVF', 'VSA', 'PLD', 'LCD/PLD', 'ρ', 'PSD', 'Qst', 'ρ(MOF/poly)', 'ρ_poly', 'FFV']].astype(float)
        #print(np.isnan(X_new).any())
        
        
        X_pred = scaler.transform(X_new)  # Transform the data
        R1 = batch_prediction(X_pred, selected_option)  
        R1_scientific = ['{:.2e}'.format(num) for num in R1]
        
        #Output the result to an Excel file
        d1 = pd.DataFrame({'pred': R1_scientific})
        newdata = pd.concat([pred_data, d1], axis=1)
        newdata.to_excel("Result/Batch_Predicted.xlsx")

        # Displays a prediction completion message
        label_P = tk.Label(root, font=('microsoft yahei', 10),
                           text='Predicted results have been stored in:\nResult/Batch_Predicted.xlsx', bg='green')
        canvas1.create_window(600, 600, window=label_P)
    else:
        #The user does not select any options and does not make predictions
        label_P = tk.Label(root, font=('microsoft yahei', 8), text='Please select an option before predicting', bg='red')
        canvas1.create_window(600, 600, window=label_P)

# Create a drop-down box
var2 = tk.StringVar()  
cm2 = ttk.Combobox(root, textvariable=var2, font=('microsoft yahei', 10))
cm2["value"] = ("TSP(CO2/N2)", "TSP(CO2/CH4)", "TSP(CO2/O2)", "TSP(CO2/H2)", "P")
canvas1.create_window(220, 600, window=cm2)

# Create prediction button
button_predict = tk.Button(root, text="Predicted", font=('microsoft yahei', 10), command=predict,relief=tk.RAISED)
#button_predict.configure(bg="lightgray",fg="black")
canvas1.create_window(380, 600, window=button_predict)

root.mainloop()