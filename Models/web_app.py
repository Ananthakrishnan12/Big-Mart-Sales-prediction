from flask import Flask,render_template,url_for,request
import pickle
import numpy as np
import joblib

app=Flask(__name__)

model_path = r'E:\Protfolio Projects\Machince Learning\A Comparative Study of Big Mart Sales Prediction\Models\RF.pickle'

model = joblib.load(
    open(model_path,'rb'))

@app.route('/')

def home():
    return render_template('home.html')


@app.route('/result',methods=['POST'])
def predict():
    
    Item_Fat_content= (request.form['Item_Fat_Content'])
    Item_Visibility= (request.form['Item_Visibility'])
    Item_MRP= (request.form['Item_MRP'])
    Outlet_Identifier= (request.form['Outlet_Identifier'])
    Outlet_Establishment_Year= (request.form['Outlet_Establishment_Year'])
    #Asthma = float(request.form['Asthma'])
    Outlet_Size= (request.form['Outlet_Size'])
    Oulet_Location_Type= (request.form['Outlet_Location_Type'])
    Outlet_Type= (request.form['Outlet_Type'])
    # grade = (request.form['grade'])
    # sqft_above = (request.form['sqft_above'])
    # sqft_basement = (request.form['sqft_basement'])
    # yr_built = (request.form['yr_built'])
    # yr_renovated=(request.form['yr_renovated'])
    # lat =(request.form['lat'])
    # sqft_living15 = (request.form['sqft_living15'])
    # sqft_lot15 = (request.form['sqft_lot15'])
    # month = (request.form['month'])
    # year = (request.form['year'])
    
    query = np.array([[Item_Fat_Content, Item_Visibility, Item_MRP, Outlet_Identifier,
       Outlet_Establishment_Year, Outlet_Size,Outlet_Location_Type,Outlet_Type]])

    prediction = model.predict(query)

    
    return render_template('result.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
