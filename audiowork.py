import SMILEapi
import csv
import pandas as pd
import os
import numpy as np
import tensorflow as tf
from const import RESULTS_ROOT_PATH, CONF

openSmile = SMILEapi.OpenSMILE()


def getLastRowData(file):
    with open(file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            last_row = row
    return last_row


def convert_to_float(element):
    try:
        return float(element)
    except ValueError:
        return element


def delete_logs(files_to_analize, results_root_path):
    csv_files_to_delete = [os.path.join(results_root_path, f"{base_name}.csv")
                           for base_name in files_to_analize]
    for file_path in csv_files_to_delete:
        if os.path.exists(file_path):
            os.remove(file_path)


def opensmile_analysis(files_to_analize, conf, audiofiles_root_path,
                       results_root_path):
    options = {}
    metadata = []
    for file in files_to_analize:
        openSmile.process(conf, {"I": audiofiles_root_path + file + '.wav',
                                 "O": results_root_path + file + '.csv'},
                          options, metadata)


def reading_csv(file_path):
    attribute_names = []
    data_section = False
    with open(file_path, 'r') as file:
        for line in file:
            if '@data' in line:
                data_section = True
                continue
            if not data_section:
                if '@attribute' in line:
                    parts = line.split()
                    if len(parts) > 1:
                        attribute_names.append(parts[1])
            else:
                row_info = line.strip()
    row_info = row_info.split(sep=',')
    info = [convert_to_float(item) for item in row_info]
    return attribute_names, info


def forming_dataframe(files_to_analize, results_root_path):
    rows = []
    for file in files_to_analize:
        attribute_names, row_info = reading_csv(results_root_path + file
                                                + '.csv')
        row_info[0] = file
        rows.append(row_info)
    df = pd.DataFrame(rows, columns=attribute_names)
    # Удалить логи?
    delete_logs(files_to_analize, results_root_path)
    return df


def get_wav_files(audiofiles_root_path):
    all_files = os.listdir(audiofiles_root_path)
    return [f[:-4] for f in all_files if f.endswith('.wav')]


def cnn_prediction(df):
    model = tf.keras.layers.TFSMLayer('model_directory',
                                      call_endpoint='serving_default')
    x = df.drop(columns=['class', 'name']).values
    x = x.reshape((x.shape[0], x.shape[1], 1))
    x = x / np.max(x)
    y_pred = model(x)
    y_pred = (y_pred['dense_5'].numpy())
    y_pred_classes = np.argmax(y_pred, axis=1)
    df['class'] = y_pred_classes
    return df


def main():
    # audiofiles_root_path = "dataset/train/"
    audiofiles_root_path = "opensmile/example-audio/"

    files_to_analize = get_wav_files(audiofiles_root_path)
    opensmile_analysis(files_to_analize, CONF, audiofiles_root_path,
                       RESULTS_ROOT_PATH)
    df = (forming_dataframe(files_to_analize, RESULTS_ROOT_PATH))
    df.to_csv('output.csv', index=False)
    cnn_prediction(df)


if __name__ == '__main__':
    main()
