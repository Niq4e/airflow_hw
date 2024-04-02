from datetime import datetime
import os
import json
import dill
import pandas as pd


path = os.path.expanduser('~/airflow_hw')


def predict():
    mod = sorted(os.listdir(f'{path}/data/models'))
    with open(f'{path}/data/models/cars_pipe.pkl', 'rb') as file:
        Model = dill.load(file)

    df_pred = pd.DataFrame(columns=['id', 'predict'])
    files_list = os.listdir(f'{path}/data/test')

    for filename in files_list:
        with open(f'{path}/data/test/{filename}', 'r') as file:
            form = json.load(file)
        data = pd.DataFrame.from_dict([form])
        prediction = Model.predict(data)

        dict_pred = {'id': data['id'].values[0], 'predict': prediction[0]}
        df = pd.DataFrame([dict_pred])
        df_pred = pd.concat([df, df_pred], ignore_index=True)

    now = datetime.now().strftime("%Y%m%d%H%M")
    df_pred.to_csv(f'{path}/data/predictions/{now}.csv', index=False)



if __name__ == '__main__':
    predict()
