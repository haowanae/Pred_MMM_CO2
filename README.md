## Usage
The software transforms the Stacking model trained to predict the CO2 performance of mixed matrix membrane materials into an interactive desktop application. 

model loading：Choose the link to download the model according to your needs, and put it in the same directory as the other folders.

[baidu model Download](https://pan.baidu.com/s/1_EdVOuzgooXxzp42gO2XXg?pwd=e64n "baidu Download")

[google model Download](https://drive.google.com/drive/folders/1yNaaLnRT4eSF_VBBlVbIf4eqNj_duuLx?usp=drive_link) 

For the convenience of users. Users do not need to install other auxiliary software, open main_MMM.exe to use.

## This software has two functions: 
1- To calculate the comprehensive separation performance (TSP) of CO2/X, where X=(N2, O2, H2, CH4), and the permeability (P) of CO2 in a single crystal material.
   A single prediction result is displayed on the interface.
   
2- Batch calculation of CO2 performance in MMM (Mixed Matrix Membrane) materials.
   The predicted result will be saved in Result/Batch_Predicted.xlsx.

## This folder includes five folders:
1- Code
     1.1Stacking_code.py that has the code for the machine learning using Stacking (for more info please visit: https://scikit- 
     learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html).  
     1.2.Predicted_MMM_CO2_code.py that has the code for a human-computer interactive interface software.

2- Extrapolation_data
     Example_MMM_6FDA-DAM.xlsx that is a sample file for batch prediction of material diffusivity.

3- Img 
     full_name.png and sample_file.png that are the interactive interface software required in the illustration picture. 
 
4- model
     The "model" folder contains both metamodels and base models that have been trained for multiple performances.
     model loading：Choose the link to download the model according to your needs, and put it in the same directory as the other folders.
     
5- Result 
     The predicted result will be automatically generated and saved in Result/Batch_Predicted.xlsx.
