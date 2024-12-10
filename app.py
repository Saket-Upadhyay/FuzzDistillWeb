import base64
import hashlib
import io
import os

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from flask import Flask, request, render_template, jsonify

from includes.constants import FN_UNIMPORTANT_FEATURES, FN_TARGET_FEATURE, FN_CPP_MEMORY_FEATURES, \
    FN_EXPLICIT_EXCLUDE_FEATURES, BB_EXPLICIT_EXCLUDE_FEATURES, BB_TARGET_FEATURE, \
    BB_UNIMPORTANT_FEATURES, BB_CPP_MEMORY_FEATURES

cache = {}

HIGH_CONFIDENCE_THRESHOLD = 0.95

app = Flask(__name__)

""" Load FN Prediction Model """
MODEL_PATH = {
    "dnnfn": os.path.join(os.path.dirname(__file__), 'models/FNpredict_TF_30_32_86.keras'),
    "dnnbb": os.path.join(os.path.dirname(__file__), 'models/BBpredict_TF_21_32_68.keras')
}

""" MAIN ROUTE """


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':

        if 'file' not in request.files:
            return render_template('index.html',
                                   message='No file part')

        file = request.files['file']
        user_modelchoice = str(request.form.get('modelselect'))
        MODEL_ACCURACY = \
            os.path.basename(MODEL_PATH.get(user_modelchoice)).strip('.keras').split('_')[-1]

        print(user_modelchoice)

        if file.filename == '':
            return render_template('index.html', message='No selected file')

        if file and allowed_file(file.filename):

            file_content = file.read()
            file.seek(0)
            file_hash = hash_file(file_content)
            cached_result = get_cached_result(file_hash)
            if cached_result is not None:
                result_df = cached_result
                print(f"USING CACHED RESULT for {file_hash}")
            else:
                result_df = get_result_from_csv(file, user_modelchoice)
                store_cached_result(file_hash, result_df)
            if user_modelchoice == 'dnnfn':
                result_df.to_csv("resultDNNdf.csv", sep=';', index=False)
            elif user_modelchoice == 'dnnbb':
                result_df.to_csv("resultDNNbb.csv", sep=';', index=False)

            all_vulnerable_functions = result_df[result_df['predictions'] == 1]

            high_confidence_df = all_vulnerable_functions[
                all_vulnerable_functions['confidence'] >= HIGH_CONFIDENCE_THRESHOLD]
            sure_vulnerable_df = all_vulnerable_functions[
                all_vulnerable_functions['confidence'] == 1]

            total_entries_to_predict = result_df.shape[0]

            all_vulnerable_functions_html, high_confidence_functions, sure_vulnerable_functions, message = get_model_prediction_result_lists(
                result_df, user_modelchoice)

            vulnerable_predictions = (result_df['predictions'] == 1).sum()
            nonvulnerable_predictions = (result_df['predictions'] == 0).sum()
            high_confidence_predictions = (
                    result_df['confidence'] > HIGH_CONFIDENCE_THRESHOLD).sum()
            sure_predictions = (result_df['confidence'] == 1).sum()
            plot1 = plot_vuln_and_non_metrics(nonvulnerable_predictions, vulnerable_predictions)
            plot2 = plot_pred_metrics(total_entries_to_predict, vulnerable_predictions,
                                      high_confidence_predictions, sure_predictions)

            return render_template('results.html',
                                   model_accuracy=MODEL_ACCURACY,
                                   predtype="DNN_BB" if user_modelchoice == 'dnnbb' else "DNN_FN",
                                   high_conf_threshold=round(HIGH_CONFIDENCE_THRESHOLD * 100),
                                   curr_file_hash=file_hash,
                                   high_confidence_functions=high_confidence_functions,
                                   sure_vulnerable_functions=sure_vulnerable_functions,
                                   total_candidates=total_entries_to_predict,
                                   vulnerable_predictions=vulnerable_predictions,
                                   high_confidence_predictions=high_confidence_predictions,
                                   sure_predictions=sure_predictions, plot1=plot1, plot2=plot2)


        else:
            return render_template('index.html', message='Allowed file types are CSV')

    if os.path.exists("resultDNNdf.csv") and os.path.isfile("resultDNNdf.csv"):
        os.remove("resultDNNdf.csv")
    if os.path.exists("resultDNNbb.csv") and os.path.isfile("resultDNNbb.csv"):
        os.remove("resultDNNbb.csv")
    return render_template('index.html', model_accuracy=-1)


""" API ROUTES"""


@app.route('/api/high-conf-list', methods=['POST'])
def api_high_confidence_list():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    usermodelchoice = str(request.form.get('modelselect'))
    file = request.files['file']
    if not usermodelchoice:
        return jsonify({'error': 'No model'}), 400
    if not file.filename:
        return jsonify({'error': 'No selected file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    file_content = file.read()
    file.seek(0)
    file_hash = hash_file(file_content)
    cached_result = get_cached_result(file_hash)
    if cached_result is not None:
        result_df = cached_result
        print(f"USING CACHED RESULT for {file_hash}")
    else:
        result_df = get_result_from_csv(file, usermodelchoice)
        store_cached_result(file_hash, result_df)

    if isinstance(result_df, str):
        return jsonify({'error': result_df}), 500

    _, high_confidence_functions, _, _ = get_model_prediction_result_lists(result_df,
                                                                           usermode=usermodelchoice)

    return jsonify(high_confidence_functions)


@app.route('/api/sure-list', methods=['POST'])
def api_sure_list():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    usermodelchoice = str(request.form.get('modelselect'))
    file = request.files['file']
    if not usermodelchoice:
        return jsonify({'error': 'No model'}), 400
    if not file.filename:
        return jsonify({'error': 'No selected file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    file_content = file.read()
    file.seek(0)
    file_hash = hash_file(file_content)
    cached_result = get_cached_result(file_hash)
    if cached_result is not None:
        result_df = cached_result
        print(f"USING CACHED RESULT for {file_hash}")
    else:
        result_df = get_result_from_csv(file, usermodelchoice)
        store_cached_result(file_hash, result_df)

    if isinstance(result_df, str):
        return jsonify({'error': result_df}), 500

    _, _, sure_vulnerable_functions, _ = get_model_prediction_result_lists(result_df,
                                                                           usermodelchoice)

    return jsonify(sure_vulnerable_functions)


@app.route('/api/all-list', methods=['POST'])
def api_all_list():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    usermodelchoice = str(request.form.get('modelselect'))
    file = request.files['file']
    if not usermodelchoice:
        return jsonify({'error': 'No model'}), 400
    if not file.filename:
        return jsonify({'error': 'No selected file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    file_content = file.read()
    file.seek(0)
    file_hash = hash_file(file_content)
    cached_result = get_cached_result(file_hash)
    if cached_result is not None:
        result_df = cached_result
        print(f"USING CACHED RESULT for {file_hash}")
    else:
        result_df = get_result_from_csv(file, usermodelchoice)
        store_cached_result(file_hash, result_df)

    if isinstance(result_df, str):
        return jsonify({'error': result_df}), 500

    all_vuln_list, _, _, _ = get_model_prediction_result_lists(result_df, usermodelchoice)

    return jsonify(all_vuln_list)


"""
curl -X POST http://your-api-url/api/clear-cache-record \
     -H "Content-Type: application/json" \
     -d '{"file_hash": "your_file_hash_here"}'
"""


@app.route('/api/clear-cache-record', methods=['GET'])
def api_clear_cache_record():
    file_hash = request.args.get('hash')
    if not file_hash:
        return jsonify({'error': 'No file hash provided'}), 400

    if file_hash in cache:
        del cache[file_hash]
        return jsonify({'message': f'Cache cleared for file hash: {file_hash}'}), 200
    else:
        return jsonify({'error': 'File hash not found in cache'}), 404


@app.route('/api/clear-cache', methods=['POST', 'GET'])
def api_clear_cache():
    cache.clear()
    return jsonify({'message': 'Entire cache cleared'}), 200


""" Data Processing Helpers"""


def plot_pred_metrics(total_candidates, vulnerable_predictions, high_confidence_predictions,
                      sure_predictions):
    plt.figure(figsize=(12, 12))

    metrics = [total_candidates, vulnerable_predictions, high_confidence_predictions,
               sure_predictions]
    labels = ['Total Candidates', 'Vulnerable Predictions', 'High Confidence Predictions',
              'Sure Predictions']

    plt.bar(labels, metrics, color=['blue', 'orange', 'green', 'red'])
    plt.xlabel('Categories')
    plt.ylabel('Count')
    plt.title('Distillation Visualization')
    plt.ylim(0, max(metrics) + 5)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    return image_base64


def plot_vuln_and_non_metrics(nonvulnerable_predictions, vulnerable_predictions):
    metrics = [nonvulnerable_predictions, vulnerable_predictions]
    labels = ['Non Vulnerable Predictions', 'Vulnerable Predictions']

    plt.figure(figsize=(12, 12))
    plt.pie(metrics, labels=labels, colors=['blue', 'orange', 'green', 'red'], autopct='%1.1f%%')
    plt.title('Prediction Summary')
    plt.axis('equal')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    return image_base64


def get_model_prediction_result_lists(df, usermode):
    if usermode == 'dnnfn':
        vulnerable_functions = df[df['predictions'] == 1]
        vulnerable_list = vulnerable_functions['Function Name'].tolist()

        high_confidence_functions = vulnerable_functions[
            vulnerable_functions['confidence'] > HIGH_CONFIDENCE_THRESHOLD]
        high_confidence_list = high_confidence_functions['Function Name'].tolist()

        sure_vulnerabilities = vulnerable_functions[vulnerable_functions['confidence'] == 1]
        sure_vulnerabilities_list = sure_vulnerabilities['Function Name'].tolist()

        message = f"Total Vulnerable Functions: {len(vulnerable_functions)}"
        return vulnerable_list, high_confidence_list, sure_vulnerabilities_list, message
    elif usermode == 'dnnbb':
        vulnerable_bbs = df[df['predictions'] == 1]
        vulnerable_list = vulnerable_bbs['Block Name'].tolist()

        high_confidence_bbs = vulnerable_bbs[
            vulnerable_bbs['confidence'] > HIGH_CONFIDENCE_THRESHOLD]
        high_confidence_list = high_confidence_bbs['Block Name'].tolist()

        sure_vulnerabilities = vulnerable_bbs[vulnerable_bbs['confidence'] == 1]
        sure_vulnerabilities_list = sure_vulnerabilities['Block Name'].tolist()

        message = f"Total Vulnerable BBS: {len(vulnerable_bbs)}"

        return vulnerable_list, high_confidence_list, sure_vulnerabilities_list, message


""" CACHING HELPERS """


def hash_file(file_content):
    sha256_hash = hashlib.sha256()
    sha256_hash.update(file_content)
    return sha256_hash.hexdigest()


def get_cached_result(file_hash):
    return cache.get(file_hash, None)


def store_cached_result(file_hash, resultdf):
    cache[file_hash] = resultdf


""" FILE HANDLING HELPERS"""


def get_result_from_csv(file, user_modelchoice):
    original_dataframe = pd.read_csv(file, sep=';')
    if user_modelchoice == 'dnnfn':
        model = tf.keras.models.load_model(MODEL_PATH.get("dnnfn"))
        features = original_dataframe.drop(
            FN_UNIMPORTANT_FEATURES + FN_TARGET_FEATURE + FN_CPP_MEMORY_FEATURES + FN_EXPLICIT_EXCLUDE_FEATURES,
            axis=1)

        predictions = model.predict(features)

        predicted_classes = (predictions > 0.5).astype(int)

        confidence_scores = predictions.flatten()
        confidence_scores_percent = (predictions.flatten() * 100).round(
            2)

        result_df = original_dataframe[
            ['Function Name']].copy()
        result_df['confidence %'] = confidence_scores_percent
        result_df['predictions'] = predicted_classes
        result_df['confidence'] = confidence_scores
    elif user_modelchoice == 'dnnbb':
        model = tf.keras.models.load_model(MODEL_PATH.get("dnnbb"))

        features = original_dataframe.drop(
            BB_UNIMPORTANT_FEATURES + BB_TARGET_FEATURE + BB_CPP_MEMORY_FEATURES + BB_EXPLICIT_EXCLUDE_FEATURES,
            axis=1)

        predictions = model.predict(features)

        predicted_classes = (predictions > 0.5).astype(int)

        confidence_scores = predictions.flatten()
        confidence_scores_percent = (predictions.flatten() * 100).round(
            2)

        result_df = original_dataframe[
            ['Block Name']].copy()
        result_df['confidence %'] = confidence_scores_percent
        result_df['predictions'] = predicted_classes
        result_df['confidence'] = confidence_scores
    else:
        return -1

    return result_df


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'


if __name__ == '__main__':
    app.run(debug=True)
