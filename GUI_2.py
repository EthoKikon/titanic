import pandas as pd
import customtkinter as ctk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier



# Feature Engineering

train_df = pd.read_csv('train.csv')

# Step 1: Impute missing ages with the mean age
mean_age = train_df['Age'].mean()
train_df['Age'] = train_df['Age'].fillna(mean_age)

# Step 2: Drop the 'Cabin' column
train_df.drop('Cabin', axis=1, inplace=True)

# Step 3: Impute missing 'Embarked' values with the most common port
most_common_embarked = train_df['Embarked'].mode()[0]
train_df['Embarked'] = train_df['Embarked'].fillna(most_common_embarked)

# Step 4: One Hot Encoding 'Sex' and 'Embarked' columns
train_df = pd.get_dummies(train_df, columns=['Sex', 'Embarked'], drop_first=True)

# Title from Name
train_df['Title'] = train_df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
train_df['Title'] = train_df['Title'].replace(['Lady', 'Countess', 'Dona'], 'Royalty')
train_df['Title'] = train_df['Title'].replace(['Mme'], 'Mrs')
train_df['Title'] = train_df['Title'].replace(['Mlle', 'Ms'], 'Miss')
train_df['Title'] = train_df['Title'].replace(['Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer'], 'Special')

# Family Size
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1

#### Creating a binary feature indicating whether the passenger is alone of with family.

# Is Alone
train_df['IsAlone'] = 0
train_df.loc[train_df['FamilySize'] == 1, 'IsAlone'] = 1

#### Grouping the ages into bins might help the model capture nonlinearities. Children and elderly passengers, for example, might have different survival probabilities compared to teenagers or adults due to factors such as priority in evacuations, physical abilities, or care requirements.

# Age Group
train_df['AgeGroup'] = pd.cut(train_df['Age'], bins=[0, 12, 18, 60, 200], labels=['Child', 'Teenager', 'Adult', 'Elderly'])

#### Calculating the fare per person, which may provide a more informative feature than the fare alone.

# Fare per Person
train_df['FarePerPerson'] = train_df['Fare'] / train_df['FamilySize']

# Convert categorical variables using one-hot encoding for Title and AgeGroup
title_dummies = pd.get_dummies(train_df['Title'], prefix='Title')
train_df = pd.concat([train_df, title_dummies], axis=1)
train_df.drop('Title', axis=1, inplace=True)
agegroup_dummies = pd.get_dummies(train_df['AgeGroup'], prefix='AgeGroup')
train_df = pd.concat([train_df, agegroup_dummies], axis=1)
train_df.drop('AgeGroup', axis=1, inplace=True)

# Step 1: Feature selection
features = ['Pclass', 
            'Age', 
            'Sex_male', 
            'Embarked_Q', 
            'Embarked_S', 
            'FamilySize', 
            'IsAlone', 
            'FarePerPerson', 
            'Title_Master', 
            'Title_Miss', 
            'Title_Mr', 
            'Title_Mrs', 
            'Title_Royalty', 
            'Title_Special', 
            'AgeGroup_Child', 
            'AgeGroup_Teenager', 
            'AgeGroup_Adult', 
            'AgeGroup_Elderly']

X = train_df[features]
y = train_df['Survived']

# Step 2: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Gradient Boosting Model (Best Performance with 83% Accuracy)
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)

   


# function to clear the data entered
def clear():
    Pclass_entry.delete(0, ctk.END)
    Name_entry.delete(0, ctk.END)
    Sex_entry.delete(0, ctk.END)
    Age_entry.delete(0, ctk.END)
    SibSp_entry.delete(0, ctk.END)
    Parch_entry.delete(0, ctk.END)
    Fare_entry.delete(0, ctk.END)
    Embarked_entry.delete(0, ctk.END)
    result_value.destroy()

   
# function to predict Titanic surviors by using the input data
def result():
        

    # values inserted by user
    Pclass_value = int(Pclass_entry.get())
    Name_value = Name_entry.get()
    Sex_value = Sex_entry.get()
    Age_value = float(Age_entry.get())
    SibSp_value = int(SibSp_entry.get())
    Parch_value = int(Parch_entry.get())
    Fare_value = float(Fare_entry.get())
    Embarked_value = Embarked_entry.get()

    user_input_df = pd.DataFrame({"Pclass": [Pclass_value], 
                                  "Name": [Name_value],
                                  "Sex": [Sex_value], 
                                  "Age": [Age_value],
                                  "SibSp":[SibSp_value], 
                                  "Parch": [Parch_value], 
                                  "Fare": [Fare_value],
                                  "Embarked": [Embarked_value]})
    
  

    # One Hot Encoding 'Sex' and 'Embarked' columns
    user_input_df = pd.get_dummies( user_input_df, columns=['Sex', 'Embarked'], drop_first=True)

    # Title from Name
    user_input_df['Title'] =   user_input_df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    user_input_df['Title'] =   user_input_df['Title'].replace(['Lady', 'Countess', 'Dona'], 'Royalty')
    user_input_df['Title'] =   user_input_df['Title'].replace(['Mme'], 'Mrs')
    user_input_df['Title'] =   user_input_df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    user_input_df['Title'] =   user_input_df['Title'].replace(['Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer'], 'Special')
    user_input_df.drop('Name', axis = 1, inplace = True)

    # Family Size
    user_input_df['FamilySize'] =  user_input_df['SibSp'] +  user_input_df['Parch'] + 1
    user_input_df.drop(columns = ['SibSp', 'Parch'], axis = 1, inplace = True)

    #### Creating a binary feature indicating whether the passenger is alone of with family.

    # Is Alone
    user_input_df['IsAlone'] = 0
    user_input_df.loc[user_input_df['FamilySize'] == 1, 'IsAlone'] = 1

    #### Grouping the ages into bins might help the model capture nonlinearities. Children and elderly passengers, for example, might have different survival probabilities compared to teenagers or adults due to factors such as priority in evacuations, physical abilities, or care requirements.

    # Age Group
    user_input_df['AgeGroup'] = pd.cut( user_input_df['Age'], bins=[0, 12, 18, 60, 200], labels=['Child', 'Teenager', 'Adult', 'Elderly'])


    #### Calculating the fare per person, which may provide a more informative feature than the fare alone.
    # Fare per Person
    user_input_df['FarePerPerson'] =  user_input_df['Fare']/user_input_df['FamilySize']
    user_input_df.drop('Fare', axis = 1, inplace = True)

    # Convert categorical variables using one-hot encoding for Title and AgeGroup
    title_dummies = pd.get_dummies(user_input_df['Title'], prefix='Title')
    user_input_df = pd.concat([ user_input_df, title_dummies], axis=1)
    user_input_df.drop('Title', axis=1, inplace=True)
    agegroup_dummies = pd.get_dummies(user_input_df['AgeGroup'], prefix='AgeGroup')
    user_input_df = pd.concat([user_input_df, agegroup_dummies], axis=1)
    user_input_df.drop('AgeGroup', axis=1, inplace=True)

    required_columns = ['Pclass', 
                        'Age', 
                        'Sex_male', 
                        'Embarked_Q', 
                        'Embarked_S', 
                        'FamilySize', 
                        'IsAlone', 
                        'FarePerPerson', 
                        'Title_Master', 
                        'Title_Miss', 
                        'Title_Mr', 
                        'Title_Mrs', 
                        'Title_Royalty', 
                        'Title_Special', 
                        'AgeGroup_Child', 
                        'AgeGroup_Teenager', 
                        'AgeGroup_Adult', 
                        'AgeGroup_Elderly']
    
    for col in required_columns:
        if col not in user_input_df.columns:
             user_input_df[col] = 0

    user_input_df = user_input_df[X.columns]   

    user_input_df = user_input_df.infer_objects(copy=False)     
    
    # starting the prediction using the input data
    prediction = gb_model.predict(user_input_df)




    # declared as global so that clear() function can access it
    global result_value

    # result of the prediction
    if prediction == 1:
        result_value = ctk.CTkLabel(window, text="RESULT: The passenger did SURVIVED!", font=ctk.CTkFont(size=20))
        result_value.grid(row=3, column=0, pady=30)
    else:
        result_value = ctk.CTkLabel(window, text="RESULT: The passenger did NOT SURVIVED!", font=ctk.CTkFont(size=20))
        result_value.grid(row=3, column=0, pady=30)

# Declaring a variable called result_value that will hold the result of prediction
result_value = None

# main window of the app
window = ctk.CTk()
window.title("Titanic Survivor Predictor")
window.geometry("620x660")
window.resizable(False, False)
ctk.set_appearance_mode("light")

# child frame of the main window
child_frame = ctk.CTkFrame(window)
child_frame.grid(row=1, column=0, padx=45, pady=(5, 10))



# labels to guide the users during data entry

Pclass_label = ctk.CTkLabel(child_frame, text="Passenger's class")
Pclass_label.grid(row=0, column=0, sticky="w")

Name_label = ctk.CTkLabel(child_frame, text="Passenger's Name")
Name_label.grid(row=1, column=0, sticky="w")

Sex_label = ctk.CTkLabel(child_frame, text="Sex")
Sex_label.grid(row=2, column=0, sticky="w")

Age_label = ctk.CTkLabel(child_frame, text="Age")
Age_label.grid(row=3, column=0, sticky="w")

SibSp_label = ctk.CTkLabel(child_frame, text="No. of Siblings")
SibSp_label.grid(row=4, column=0, sticky="w")

Parch_label = ctk.CTkLabel(child_frame, text="No. of Parents/Children")
Parch_label.grid(row=5, column=0, sticky="w")

Fare_label = ctk.CTkLabel(child_frame, text="Ticket Fare")
Fare_label.grid(row=6, column=0, sticky="w")

Embarked_label = ctk.CTkLabel(child_frame, text="Embarked")
Embarked_label.grid(row=7, column=0, sticky="w")



# entries to receive data from users
Pclass_entry = ctk.CTkEntry(child_frame)
Pclass_entry.grid(row=0, column=1)

Name_entry = ctk.CTkEntry(child_frame)
Name_entry.grid(row=1, column=1)

Sex_entry = ctk.CTkEntry(child_frame)
Sex_entry.grid(row=2, column=1)

Age_entry = ctk.CTkEntry(child_frame)
Age_entry.grid(row=3, column=1)

SibSp_entry = ctk.CTkEntry(child_frame)
SibSp_entry.grid(row=4, column=1)

Parch_entry = ctk.CTkEntry(child_frame)
Parch_entry.grid(row=5, column=1)

Fare_entry = ctk.CTkEntry(child_frame)
Fare_entry.grid(row=6, column=1)

Embarked_entry = ctk.CTkEntry(child_frame)
Embarked_entry.grid(row=7, column=1)



# rearranging the widgets of child frame
for widget in child_frame.winfo_children():
    widget.grid_configure(padx=80, pady=25)


# Buttons to perform prediction or deletion of input data
predict_button = ctk.CTkButton(window, text="PREDICT", command=result, hover=True, hover_color="blue")
predict_button.grid(row=2, column=0, padx=(0, 200))

clear_button = ctk.CTkButton(window, text="CLEAR", command=clear, hover=True, hover_color="blue")
clear_button.grid(row=2, column=0, padx=(200, 0))

window.mainloop()


