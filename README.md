## BIG Mart Sales Prediction:

Abstract: Currently, supermarket run-centres, Big Marts keep track of each individual item's sales data in order to anticipate potential consumer demand and update inventory management. Anomalies and general trends are often discovered by mining the data warehouse's data store. For retailers like Big Mart, the resulting data can be used to forecast future sales volume using various machine learning techniques like big mart. A predictive model was developed using Xgboost, Linear regression, Polynomial regression, and Ridge regression techniques for forecasting the sales of a business such as Big-Mart, and it was discovered that the model outperforms existing models.

Installation: 
Anaconda 2021 Required Libaries: numpy==1.20.3 Pandas==1.3.5  matplotlib==3.4.3 scikit-learn==0.24.1

Data collection: https://www.kaggle.com/datasets/shivan118/big-mart-sales-prediction-datasets

Data Descriptions:
Item_Identifier ----- Unique product ID

Item_Weight ---- Weight of product

Item_Fat_Content ----- Whether the product is low fat or not

Item_Visibility ---- The % of the total display area of all products in a store allocated to the particular product

Item_Type ---- The category to which the product belongs

Item_MRP ----- Maximum Retail Price (list price) of the product

Outlet_Identifier ----- Unique store ID

Outlet_Establishment_Year ----- The year in which store store was established

Outlet_Size ----- The size of the store in terms of ground area covered

Outlet_Location_Type ---- The type of city in which the store is located

Outlet_Type ---- whether the outlet is just a grocery store or some sort of supermarket

steps involved in this Model:
1.Data Preprocessing.
2.EDA
3.Feature Engineering
4.Feature scaling
5.Model selection
6.Model Training (Regression Models)
7.Model Testing
8.Performance metrics
9.Model Deployment (Flask)