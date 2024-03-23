import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV

# Load the train file
train_data = pd.read_csv('train.csv')

X_train = train_data[['Property_Type', 'Property_Area', 'Number_of_Windows', 'Number_of_Doors', 'Furnishing', 'Frequency_of_Powercuts', 'Power_Backup', 'Water_Supply', 'Traffic_Density_Score', 'Crime_Rate', 'Dust_and_Noise', 'Air_Quality_Index', 'Neighborhood_Review']]
y_train = train_data['Habitability_score']

X_train = pd.get_dummies(X_train)

imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20],
    'num_leaves': [31, 50, 70]
}

lgbm_model = LGBMRegressor()
grid_search = GridSearchCV(lgbm_model, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train_imputed, y_train)

best_lgbm_model = grid_search.best_estimator_

# Ask user for the test CSV file path
test_file_path = input("Enter the path to the test CSV file: ")

test_data = pd.read_csv(test_file_path)

# Preprocess the test data
X_test = pd.get_dummies(test_data)
missing_cols = set(X_train.columns) - set(X_test.columns)
for col in missing_cols:
    X_test[col] = 0


X_test = X_test[X_train.columns]

# missing values dekho
X_test_imputed = imputer.transform(X_test)

# prediction karo
predictions = best_lgbm_model.predict(X_test_imputed)

# user input dega 
output_choice = input("Do you want the entire result ('all'), habitability scores for the whole file ('whole'), or a specific Property_ID ('specific')? ")

if output_choice.lower() == 'all':
    print("Habitability Scores for the Test Data:")
    for index, prediction in enumerate(predictions):
        print(f"Property_ID: {test_data['Property_ID'][index]} - Habitability Score: {prediction}")
elif output_choice.lower() == 'whole':
    for index, prediction in enumerate(predictions):
        print(f"Property_ID: {test_data['Property_ID'][index]} - Habitability Score: {prediction}")
elif output_choice.lower() == 'specific':
    property_id = input("Enter the Property_ID you want to check: ")
    specific_property_index = test_data[test_data['Property_ID'] == property_id].index
    if not specific_property_index.empty:
        specific_property_prediction = predictions[specific_property_index[0]]
        print(f"Habitability Score for Property_ID {property_id}: {specific_property_prediction}")
    else:
        print("Property_ID not found in the test data.")
