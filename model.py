from google.colab import files
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

train_data = pd.read_csv('train.csv')

X_train = train_data[['Property_Type', 'Property_Area', 'Number_of_Windows', 'Number_of_Doors', 'Furnishing', 'Frequency_of_Powercuts', 'Power_Backup', 'Water_Supply', 'Traffic_Density_Score', 'Crime_Rate', 'Dust_and_Noise', 'Air_Quality_Index', 'Neighborhood_Review']]
y_train = train_data['Habitability_score']

X_train = pd.get_dummies(X_train)

imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)

rf_model = RandomForestRegressor()
rf_model.fit(X_train_imputed, y_train)

uploaded = files.upload()

for file_name in uploaded.keys():
    test_data = pd.read_csv(file_name)

    X_test = test_data[['Property_Type', 'Property_Area', 'Number_of_Windows', 'Number_of_Doors', 'Furnishing', 'Frequency_of_Powercuts', 'Power_Backup', 'Water_Supply', 'Traffic_Density_Score', 'Crime_Rate', 'Dust_and_Noise', 'Air_Quality_Index', 'Neighborhood_Review']]
    X_test = pd.get_dummies(X_test)
    X_test_imputed = imputer.transform(X_test)

    predictions = rf_model.predict(X_test_imputed)

    print('Habitability Score Predictions:')
    print(predictions)
