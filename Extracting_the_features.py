import librosa
import librosa.display
import numpy as np
from mysql.connector import connect, Error
import os
from datetime import date


'''bring the sound names from the DB so you can use one of them/create new to specify predictions type'''
try:
    with connect(
            host="localhost",
            user=os.environ.get('MYSQL_USER'),
            password=os.environ.get('MYSQL_PASSWORD'),
            database="sound_recognition",
    ) as connection:
        cursor = connection.cursor()
        cursor.execute('SELECT sound_name, prediction_type FROM sound_names')
        result = cursor.fetchall()
        connection.commit()

except Error as e:
    print(e)

sound_names_received = [x[0] for x in result]
prediction_types_received = [x[1] for x in result]
sound_names = ', '.join(sound_names_received)

def extract_features(signal):
    return [
        librosa.feature.zero_crossing_rate(signal)[0,0],
        librosa.stft(signal)[0,0],
        librosa.feature.spectral_centroid(signal)[0,0],
    ]

#loading the audio
audio_path = ''
audio_signal, sr = librosa.load(audio_path)

#getting the onsets
hop_length = 512
onset_samples = librosa.onset.onset_detect(y=audio_signal, sr=sr, units='samples', hop_length=hop_length,
                                           backtrack=True)

#separating signals by onsets (shortest onset length)
event_duration = min(np.diff(onset_samples))

separated_signals=[]
for i in range(len(onset_samples)):
    separated_signals.append(audio_signal[onset_samples[i]:onset_samples[i] + event_duration])

#extracting the features
all_signal_features = np.array([extract_features(x) for x in separated_signals])
zcr = np.array([x[0] for x in all_signal_features])
stft = np.array([x[1] for x in all_signal_features])
spectral_centroid = np.array([x[2] for x in all_signal_features])

#define data type
sound_name = input('{} \n Above is the list of existing sound types. Pick one of them to label inserted data, or create\
a new sound id by typing it\'s name:\n '.format(sound_names))

predictions_type = 0
if sound_name in sound_names_received:
    predictions_type = prediction_types_received[sound_names_received.index(sound_name)]
else:
    predictions_type = prediction_types_received[-1] + 1
    try:
        with connect(
                host="localhost",
                user=os.environ.get('MYSQL_USER'),
                password=os.environ.get('MYSQL_PASSWORD'),
                database="sound_recognition",
        ) as connection:
            cursor = connection.cursor()
            cursor.execute('INSERT INTO sound_names (sound_name, prediction_type) VALUES (%s, %s)',
                           [sound_name, predictions_type])
            connection.commit()

    except Error as e:
        print(e)

predictions_table = [int(predictions_type)] * len(separated_signals)

#conversions so you can insert data to MySQL
conv_zcr = [x.tobytes() for x in zcr]
conv_stft = [x.tobytes() for x in stft]
conv_spectral_centroid = [x.tobytes() for x in spectral_centroid]
conv_event_duration = event_duration.item()

#feeding the DB
data = []
for x in range(len(predictions_table)):
    data.append((conv_zcr[x], conv_stft[x], conv_spectral_centroid[x], predictions_table[x], date.today(), conv_event_duration))

insert_new_features_query = '''
    INSERT INTO sound_dataset
    (zcr, stft, spectral_centroid, prediction, insert_date, event_duration)
    VALUES (%s, %s, %s, %s, %s, %s)'''
try:
    with connect(
            host="localhost",
            user=os.environ.get('MYSQL_USER'),
            password=os.environ.get('MYSQL_PASSWORD'),
            database="sound_recognition",
    ) as connection:
        cursor = connection.cursor()
        cursor.executemany(insert_new_features_query, data)
        connection.commit()

except Error as e:
    print(e)



