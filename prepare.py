import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_data():
    '''Returns the clean version of the dataset.'''
    filename = 'exercise_dataset.csv'
    cleaned_data = 'clean_ex_data.csv'
    if os.path.exists(cleaned_data):
        df = pd.read_csv(cleaned_data, index_col=0)
        print('Data found.')
        return df
    else:
        df = pd.read_csv(filename)
        df.drop(columns=['Gender', 'Water_Intake (liters)', 'Workout_Type', 'BMI',
                         'Workout_Frequency (days/week)', 'Experience_Level'], inplace=True)
        df.columns = [col.lower() for col in df.columns]
        df.to_csv(cleaned_data)
        print('Data saved.')
        return df

def split_data(df):
    '''Splits data into train, validate, and test subsets.'''
    seed = 42
    train, val_test = train_test_split(df, train_size=0.7, random_state=seed)
    val, test = train_test_split(val_test, train_size=0.5, random_state=seed)
    print('Data split.')
    return train, val, test

def scale_data(train, val, test):
    '''Scales numerical data using Standard Scaler.'''
    to_scale = [col for col in train.columns if col != 'calories_burned']
    ss = StandardScaler()
    train[to_scale] = ss.fit_transform(train[to_scale])
    val[to_scale] = ss.transform(val[to_scale])
    test[to_scale] = ss.transform(test[to_scale])
    print('Data scaled.')
    return train, val, test

def xy_split(train, val, test):
    '''Splits the subsets into X and y.'''
    targ = 'calories_burned'
    X_train = train.drop(columns=[targ])
    y_train = train[targ]
    X_val = val.drop(columns=[targ])
    y_val = val[targ]
    X_test = test.drop(columns=[targ])
    y_test = test[targ]
    print('X and y created.')
    return X_train, y_train, X_val, y_val, X_test, y_test