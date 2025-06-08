import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

data, meta = arff.loadarff('data.arff')  

df = pd.DataFrame(data)

# Cây quyết định
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].apply(lambda x: x.decode('utf-8').strip())

label_encoders = {}
for column in df.columns:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

X = df.drop(columns='violence_type') 
y = df['violence_type']              

clf = DecisionTreeClassifier(criterion='entropy')  
clf.fit(X, y)

plt.figure(figsize=(15,5))
plot_tree(clf, 
          feature_names=X.columns,        
          class_names=label_encoders['violence_type'].classes_, 
          filled=False,                     
          rounded=False,                   
          label='none',                   
          impurity=True,                 
          proportion=False)               
plt.show()

# Test
def predict_violence_type():
    print("Enter the following information:")
    
    gender = input(f"Gender (Enter one: {list(label_encoders['gender'].classes_)}): ")
    year = input(f"Year (Enter a value from 2013 to 2023): ")  
    incomegroup = input(f"Income Group (Choose one: {list(label_encoders['incomegroup'].classes_)}): ")
    bully_group = input(f"Bully Group (Choose one: {list(label_encoders['bully_group'].classes_)}): ")
    age_group = input(f"Age Group (Choose one: {list(label_encoders['age_group'].classes_)}): ")

    user_input = {
        'gender': [gender],
        'year': [year],      
        'incomegroup': [incomegroup],
        'bully_group': [bully_group],
        'age_group': [age_group]
    }

    user_df = pd.DataFrame(user_input)

    try:
        user_df = user_df[X.columns]
    except KeyError as e:
        print("Error: Unrecognized feature input.")
        print(f"Unrecognized label: {e}")
        print("Unpredictable")
        return

    for column in user_df.columns:
        if column in label_encoders:
            try:
                user_df[column] = label_encoders[column].transform(user_df[column])
            except ValueError:
                print(f"Unrecognized label in column '{column}'. Please provide a valid value.")
                print("Unpredictable")
                return

    try:
        prediction = clf.predict(user_df)
        predicted_class = label_encoders['violence_type'].inverse_transform(prediction)
        print(f"Predicted violence type: {predicted_class[0]}")
    except Exception as e:
        print("Unpredictable due to unexpected error:", e)

predict_violence_type()


