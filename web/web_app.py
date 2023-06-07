from flask import Flask,render_template,url_for,request
import pickle
import numpy as np

app=Flask(__name__)

model_path = r'E:\Protfolio Projects\Machince Learning\A Comparative Study of Big Mart Sales Prediction\model\RF.pickle'
model = pickle.load(
    open(model_path, 'rb'))

@app.route('/')

def home():
    return render_template('home.html')

@app.route('/result',methods=['POST'])
def predict():
    # Getting the data from the form
    Item_Fat_Content = float(request.form['Item_Fat_Content'])
    Item_Visibility = float(request.form['Item_Visibility'])
    Item_MRP = float(request.form['Item_MRP'])
    Outlet_Identifier = float(request.form['Outlet_Identifier'])
    Outlet_Establishment_Year = int(request.form['Outlet_Establishment_Year'])
    Outlet_Size = float(request.form['Outlet_Size'])
    Outlet_Location_Type = float(request.form['Outlet_Location_Type'])
    Outlet_Type = float(request.form['Outlet_Type'])

    query = np.array([[Item_Fat_Content, Item_Visibility, Item_MRP, Outlet_Identifier,
       Outlet_Establishment_Year, Outlet_Size, Outlet_Location_Type,
       Outlet_Type]])

    prediction = model.predict(query)
    print(prediction)

    
    return render_template('result.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
