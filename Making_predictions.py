import numpy as np
import sklearn
from mysql.connector import connect, Error
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

'''MySQL connection and data receiving'''
def get_from_DB(query):
    try:
        with connect(
                host="localhost",
                user=os.environ.get('MYSQL_USER'),
                password=os.environ.get('MYSQL_PASSWORD'),
                database="sound_recognition",
        ) as connection:
            cursor = connection.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            connection.commit()

    except Error as e:
        print(e)
    return result

#getting sound names from the DB
sound_names_query = 'SELECT sound_name, prediction_type FROM sound_names'
sound_names_result = get_from_DB(sound_names_query)

sound_names_received = [x[0] for x in sound_names_result]
all_ids_received = [x[1] for x in sound_names_result]

#getting feature names from DB
feature_names_query = '''SELECT COLUMN_NAME
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = 'sound_recognition' AND TABLE_NAME = 'sound_dataset'
    AND COLUMN_NAME NOT IN ('insert_date', 'event_duration', 'prediction')
    '''
feature_names_result = get_from_DB(feature_names_query)
feature_names_received = [x[0] for x in feature_names_result]

#asking the user what kind of data to get from DB
data_choice = [str(x) for x in input('{} \n Above is the list of existing sound types. Type in names of which you want to use.\
Separate them with a comma followed by a space. Insert names in the exact same order as they are listed.\
 (ex. sound1, sound2, sound5, ...):\n '.format(', '.join(sound_names_received))).split(', ')]

feature_choice = input('These are available features you can use: {} \n Choose the ones you want to use in your dataset.\
Separate them with a comma followed by a space (ex. feature1, feature2, ...): \n'.format(', '.join(feature_names_received)))

#getting the amount number of data available to use
data_amount_query = '''SELECT sn.sound_name, prediction, COUNT(*)
    FROM sound_dataset INNER JOIN sound_names sn
    ON prediction = sn.prediction_type
    WHERE sn.sound_name IN {}
    GROUP BY prediction'''.format(tuple(data_choice))

#data_amount_result shape: [(sound name, prediction, number of predictions)] ex. [('cowbell', 6, 402), ('clap_stack', 15, 36)...]
data_amount_result = get_from_DB(data_amount_query)
data_amount_choice = []
for x in data_amount_result:
    amount_number = input('There are {} datapoints available for {}. Please type in how many of these you want to use.\n'
                          .format(x[2], x[0]))
    data_amount_choice.append('(prediction = {} AND RN <= {})'.format((x[1]), amount_number))

#getting the final dataset
data_choice_query = '''SELECT {}, prediction
    FROM (SELECT {}, prediction,
    ROW_NUMBER() OVER(PARTITION BY prediction) as RN
    FROM sound_dataset ds INNER JOIN sound_names sn
    ON prediction = sn.prediction_type
    WHERE sn.sound_name IN {}) sub
    WHERE {}'''.format(feature_choice, feature_choice, tuple(data_choice), ' OR '.join(data_amount_choice))

all_data = get_from_DB(data_choice_query)

'''Reconversion of data received from MySQL'''
#the shape of data received is [(feature1, feature2, ... ,prediction), (feature1, feature2, ..., prediction) ...]

predictions = [x[-1] for x in all_data]

features = np.array([])
for x in all_data:
    datapoint = np.array([])
    for y in x[:-1]:
        datapoint = np.append(datapoint, np.frombuffer(y, dtype='complex128'))
    features = np.append(features, datapoint)

number_of_features = len(all_data[0]) - 1
number_of_datapoints = len(all_data)
features = features.reshape(number_of_datapoints, number_of_features)

'''Calculating the features'''

features_abs = np.abs(features) ** 2

scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1,1))
scaled_features = scaler.fit_transform(features_abs)

'''Applying knn to the dataset'''

X_train, X_test, y_train, y_test = train_test_split(features_abs, predictions, random_state=0)

clf = KNeighborsClassifier(n_neighbors=4)
clf.fit(X_train, y_train)

print('Test set accuracy for knnClassifier: {:.2f}'.format(clf.score(X_test, y_test)))
