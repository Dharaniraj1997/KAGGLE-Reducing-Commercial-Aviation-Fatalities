# Flask Imports
import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

# Other Imports
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import warnings
import scipy.signal as signal

warnings.filterwarnings("ignore")
UPLOAD_FOLDER = './uploads'
PREDICTIONS_FOLDER = './predictions'
ALLOWED_EXTENSIONS = {'csv'}

# APP Configuration
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREDICTIONS_FOLDER'] = PREDICTIONS_FOLDER


def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024 ** 2
    print("Memory usage of properties dataframe is :", start_mem_usg, " MB")
    NAlist = []  # Keeps track of columns that have missing values filled in.
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings

            # Print current column type
            # print("******************************")
            # print("Column: ",col)
            # print("dtype before: ",props[col].dtype)

            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()

            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all():
                NAlist.append(col)
                props[col].fillna(mn - 1, inplace=True)

                # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if -0.01 < result < 0.01:
                IsInt = True

            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)

                        # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)

            # Print new column type
            # print("dtype after: ",props[col].dtype)
            # print("******************************")

    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024 ** 2
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return props


def noise_removal(noisy_data, Wn):
    N = 3
    B, A = signal.butter(N, Wn)
    return signal.filtfilt(B, A, noisy_data)


def RCAF(x):
    print('Processing: ')

    print("To reduce RAM usage")
    x = reduce_mem_usage(x)

    x.sort_values(["crew", "time"], ascending=True).groupby("experiment")

    # w = 0.1  # Noise Removal in ECG DATA
    # x['smoothened_ecg_data'] = noise_removal(x["ecg"], w)
    #
    # w = 0.7  # Noise Removal in GSR data
    # x['smoothened_gsr_data'] = noise_removal(x["gsr"], w)
    #
    # w = 0.7  # Noise Removal in Respiration data
    # x['smoothened_r_data'] = noise_removal(x["r"], w)
    #
    # w = 0.7  # Noise Removal in EEG data in TRAIN DATA
    # feats = ["eeg_fp1", "eeg_f7", "eeg_f8", "eeg_t4", "eeg_t6", "eeg_t5", "eeg_t3", "eeg_fp2", "eeg_o1", "eeg_p3",
    #          "eeg_pz", "eeg_f3", "eeg_fz", "eeg_f4", "eeg_c4", "eeg_p4", "eeg_poz", "eeg_c3", "eeg_cz", "eeg_o2"]
    # x['smoothened_eeg_fp1'] = noise_removal(x['eeg_fp1'], w)  # Adding the smoothened data to the x dataset
    # x['smoothened_eeg_f7'] = noise_removal(x['eeg_f7'], w)
    # x['smoothened_eeg_f8'] = noise_removal(x['eeg_f8'], w)
    # x['smoothened_eeg_t4'] = noise_removal(x['eeg_t4'], w)
    # x['smoothened_eeg_t6'] = noise_removal(x['eeg_t6'], w)
    # x['smoothened_eeg_t5'] = noise_removal(x['eeg_t5'], w)
    # x['smoothened_eeg_t3'] = noise_removal(x['eeg_t3'], w)
    # x['smoothened_eeg_fp2'] = noise_removal(x['eeg_fp2'], w)
    # x['smoothened_eeg_o1'] = noise_removal(x['eeg_o1'], w)
    # x['smoothened_eeg_p3'] = noise_removal(x['eeg_p3'], w)
    # x['smoothened_eeg_pz'] = noise_removal(x['eeg_pz'], w)
    # x['smoothened_eeg_f3'] = noise_removal(x['eeg_f3'], w)
    # x['smoothened_eeg_fz'] = noise_removal(x['eeg_fz'], w)
    # x['smoothened_eeg_f4'] = noise_removal(x['eeg_f4'], w)
    # x['smoothened_eeg_c4'] = noise_removal(x['eeg_c4'], w)
    # x['smoothened_eeg_p4'] = noise_removal(x['eeg_p4'], w)
    # x['smoothened_eeg_poz'] = noise_removal(x['eeg_poz'], w)
    # x['smoothened_eeg_c3'] = noise_removal(x['eeg_c3'], w)
    # x['smoothened_eeg_cz'] = noise_removal(x['eeg_cz'], w)
    # x['smoothened_eeg_o2'] = noise_removal(x['eeg_o2'], w)

    #Normalizing the EEG features in TRAIN DATASET
    # scaler = MinMaxScaler()
    # x[['smoothened_eeg_fp1']] = scaler.fit_transform(x[['smoothened_eeg_fp1']])
    # x[['smoothened_eeg_f7']] = scaler.fit_transform(x[['smoothened_eeg_f7']])
    # x[['smoothened_eeg_f8']] = scaler.fit_transform(x[['smoothened_eeg_f8']])
    # x[['smoothened_eeg_t6']] = scaler.fit_transform(x[['smoothened_eeg_t6']])
    # x[['smoothened_eeg_t4']] = scaler.fit_transform(x[['smoothened_eeg_t4']])
    # x[['smoothened_eeg_t5']] = scaler.fit_transform(x[['smoothened_eeg_t5']])
    # x[['smoothened_eeg_t3']] = scaler.fit_transform(x[['smoothened_eeg_t3']])
    # x[['smoothened_eeg_fp2']] = scaler.fit_transform(x[['smoothened_eeg_fp2']])
    # x[['smoothened_eeg_o1']] = scaler.fit_transform(x[['smoothened_eeg_o1']])
    # x[['smoothened_eeg_p3']] = scaler.fit_transform(x[['smoothened_eeg_p3']])
    # x[['smoothened_eeg_pz']] = scaler.fit_transform(x[['smoothened_eeg_pz']])
    # x[['smoothened_eeg_f3']] = scaler.fit_transform(x[['smoothened_eeg_f3']])
    # x[['smoothened_eeg_fz']] = scaler.fit_transform(x[['smoothened_eeg_fz']])
    # x[['smoothened_eeg_f4']] = scaler.fit_transform(x[['smoothened_eeg_f4']])
    # x[['smoothened_eeg_c4']] = scaler.fit_transform(x[['smoothened_eeg_c4']])
    # x[['smoothened_eeg_p4']] = scaler.fit_transform(x[['smoothened_eeg_p4']])
    # x[['smoothened_eeg_poz']] = scaler.fit_transform(x[['smoothened_eeg_poz']])
    # x[['smoothened_eeg_c3']] = scaler.fit_transform(x[['smoothened_eeg_c3']])
    # x[['smoothened_eeg_cz']] = scaler.fit_transform(x[['smoothened_eeg_cz']])
    # x[['smoothened_eeg_o2']] = scaler.fit_transform(x[['smoothened_eeg_o2']])
    #
    # # Normalizing the SMOOTHENED ECG data of TRAIN AND TEST DATASET
    # x[['smoothened_ecg_data']] = scaler.fit_transform(x[['smoothened_ecg_data']])
    #
    # # Normalizing the SMOOTHENED R data of TRAIN AND TEST DATASET
    # x[['smoothened_r_data']] = scaler.fit_transform(x[['smoothened_r_data']])
    #
    # # Normalizing the SMOOTHENED GSR data of TRAIN AND TEST DATASET
    # x[['smoothened_gsr_data']] = scaler.fit_transform(x[['smoothened_gsr_data']])

    # Normalizing the EEG features in TRAIN DATASET
    scaler = MinMaxScaler()
    x[['eeg_fp1']] = scaler.fit_transform(x[['eeg_fp1']])
    x[['eeg_f7']] = scaler.fit_transform(x[['eeg_f7']])
    x[['eeg_f8']] = scaler.fit_transform(x[['eeg_f8']])
    x[['eeg_t6']] = scaler.fit_transform(x[['eeg_t6']])
    x[['eeg_t4']] = scaler.fit_transform(x[['eeg_t4']])
    x[['eeg_t5']] = scaler.fit_transform(x[['eeg_t5']])
    x[['eeg_t3']] = scaler.fit_transform(x[['eeg_t3']])
    x[['eeg_fp2']] = scaler.fit_transform(x[['eeg_fp2']])
    x[['eeg_o1']] = scaler.fit_transform(x[['eeg_o1']])
    x[['eeg_p3']] = scaler.fit_transform(x[['eeg_p3']])
    x[['eeg_pz']] = scaler.fit_transform(x[['eeg_pz']])
    x[['eeg_f3']] = scaler.fit_transform(x[['eeg_f3']])
    x[['eeg_fz']] = scaler.fit_transform(x[['eeg_fz']])
    x[['eeg_f4']] = scaler.fit_transform(x[['eeg_f4']])
    x[['eeg_c4']] = scaler.fit_transform(x[['eeg_c4']])
    x[['eeg_p4']] = scaler.fit_transform(x[['eeg_p4']])
    x[['eeg_poz']] = scaler.fit_transform(x[['eeg_poz']])
    x[['eeg_c3']] = scaler.fit_transform(x[['eeg_c3']])
    x[['eeg_cz']] = scaler.fit_transform(x[['eeg_cz']])
    x[['eeg_o2']] = scaler.fit_transform(x[['eeg_o2']])

    # # Normalizing the SMOOTHENED ECG data of TRAIN AND TEST DATASET
    x[['ecg']] = scaler.fit_transform(x[['ecg']])

    # # Normalizing the SMOOTHENED R data of TRAIN AND TEST DATASET
    x[['r']] = scaler.fit_transform(x[['r']])

    # # Normalizing the SMOOTHENED GSR data of TRAIN AND TEST DATASET
    x[['gsr']] = scaler.fit_transform(x[['gsr']])

    # Dropping features
    # x = x.drop(
    #     ["eeg_fp1", "eeg_f7", "eeg_f8", "eeg_t4", "eeg_t6", "eeg_t5", "eeg_t3", "eeg_o1", "eeg_p3", "eeg_pz", "eeg_f3",
    #      "eeg_fz", "eeg_f4", "eeg_c4", "eeg_p4", "eeg_poz", "eeg_c3", "eeg_cz", "eeg_o2", "r", "gsr", "ecg", "eeg_fp2"],
    #     axis=1)

    # Number of features
    print(x.shape)

    # EXPERIMENT feature is in Alphabets,so we convert it into numericals
    x['experiment'] = x['experiment'].map({'CA': 0, 'DA': 1, 'SS': 2, 'LOFT': 3})
    x["experiment"] = x["experiment"].astype('int8')

    # Model = Light GBM ( Light Gradient Boosting Machine)

    # Exporting the trained model. (To avoid training again and again)
    # joblib.dump(model_lgb,'final_model.pkl')

    # Importing the pretrained model
    infile = open('Final_model.pkl', 'rb')
    model_lgb = pickle.load(infile)

    # Deploying the model
    predicted_lgb_kaggle = model_lgb.predict(x, num_iteration=model_lgb.best_iteration,
                                             predict_disable_shape_check=True)

    # Saving the ids for further merging of features in csv.
    df_sub = pd.DataFrame()
    df_sub['id'] = x['id']
    x = x.drop('id', axis=1)

    df_sub = pd.DataFrame(np.concatenate((np.arange(len(x))[:, np.newaxis], predicted_lgb_kaggle), axis=1),
                          columns=['id', 'A', 'B', 'C', 'D'])

    # Creating the submission csv file.
    df_sub['id'] = df_sub['id'].astype(int)

    df_sub.to_csv(PREDICTIONS_FOLDER + "\\predictions.csv", index=False)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Save file to uploads directory
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            temp = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            RCAF(temp)
            return redirect(url_for('predictions_file', filename="predictions.csv"))
    return render_template('upload.html')


@app.route('/uploads/<filename>')
def predictions_file(filename):
    return send_from_directory(app.config['PREDICTIONS_FOLDER'],
                               filename)


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=2002, debug=True)
