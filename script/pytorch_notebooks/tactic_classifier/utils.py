import pandas as pd
# from transformers import BertTokenizer
from torch import nn
import torch,math,logging,os
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import numpy as np
import time
import datetime
from sklearn import preprocessing

import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import tensorflow as tf
from google.cloud import storage

import os, sys
from zipfile import ZipFile
import json
import collections


def get_bucket_name():
    return "bert_tutorial"


def get_bucket_path():
    return "gs://{}".format(get_bucket_name())


def get_storage_client_project():
    return "cloud_tpu_tutorial"


def get_experiment_dir(experiment):
    return 'data/reviewed_data/experiment/{}'.format(experiment)


def get_tactic_dir(experiment, tactic):
    return "{}/{}".format(get_experiment_dir(experiment), tactic.replace(" ", "_"))


def get_report_dir(experiment, mean_pooling_parameters, hidden_layer_used, num_of_hidden_units):
    return '{}/reports/{}/{}'.format(get_experiment_dir(experiment),
                                     get_mean_pooling_path_variable(mean_pooling_parameters),
                                     get_hidden_layer_path_variable(hidden_layer_used, num_of_hidden_units))


def get_storage_bucket():
    storage_client = storage.Client(project=get_storage_client_project())
    return storage_client.get_bucket(get_bucket_name())


def download_tactic_data(tactic, experiment):
    bucket_path = get_bucket_path()
    tactic_dir = get_tactic_dir(experiment, tactic)
    print("{}/{}/code_snippets.csv".format(bucket_path, tactic_dir))
    return pd.read_csv("{}/{}/code_snippets.csv".format(bucket_path, tactic_dir))


def tokenize_input_text(documents: list, tokenizer, max_input_length=512, use_leading_tokens = True):
    """
    Returns a len(documents) * max_sequences_per_document * 3 * 512 tensor where len(documents) is the batch
     dimension and the others encode bert input.

     This is the input to any of the document bert architectures.

     :param documents: a list of text documents
     :param tokenizer: the sentence piece bert tokenizer
    :param max_input_length:
    :param use_leading: uses the leading max_input_length number of tokens from input text
    :return:
    """
    tokenized_documents = [tokenizer.tokenize(document) for document in
                           documents]
    if use_leading_tokens:
        max_sequences_per_document = 1
    else:
        max_sequences_per_document = math.ceil(
            max(len(x) / (max_input_length - 2) for x in tokenized_documents))
        assert max_sequences_per_document <= 20, "Your document is to large, arbitrary size when writing"

    output = torch.zeros(
        size=(len(documents), max_sequences_per_document, 3, 512),
        dtype=torch.long)
    document_seq_lengths = []  # number of sequence generated per document
    # Need to use 510 to account for 2 padding tokens
    for doc_index, tokenized_document in enumerate(tokenized_documents):
        max_seq_index = 0
        for seq_index, i in enumerate(
                range(0, len(tokenized_document), (max_input_length - 2))):
            raw_tokens = tokenized_document[i:i + (max_input_length - 2)]
            tokens = []
            input_type_ids = []

            tokens.append("[CLS]")
            input_type_ids.append(0)
            for token in raw_tokens:
                tokens.append(token)
                input_type_ids.append(0)
            tokens.append("[SEP]")
            input_type_ids.append(0)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            attention_masks = [1] * len(input_ids)

            while len(input_ids) < max_input_length:
                input_ids.append(0)
                input_type_ids.append(0)
                attention_masks.append(0)

            assert len(input_ids) == 512 and len(
                attention_masks) == 512 and len(input_type_ids) == 512

            # we are ready to rumble
            output[doc_index][seq_index] = torch.cat(
                (torch.LongTensor(input_ids).unsqueeze(0),
                 torch.LongTensor(input_type_ids).unsqueeze(0),
                 torch.LongTensor(attention_masks).unsqueeze(0)),
                dim=0)
            max_seq_index = seq_index
            if use_leading_tokens:
                break
        document_seq_lengths.append(max_seq_index + 1)

    return output, torch.LongTensor(document_seq_lengths)


def create_data_loader(dataset, sampler, batch_size):
    return DataLoader(
                dataset,  # The samples.
                sampler = sampler(dataset),  # Select batches
                batch_size = batch_size  # batch size.
            )


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def get_label_encoder(labels):
  le = preprocessing.LabelEncoder()
  le.fit(labels)
  return le


def to(temp, device):
   return temp.to(device)


# Plot and save the confusion matrix
#Refer : https://stackoverflow.com/a/50386871/3215142
def plot_and_save_confusion_matrix(file_name,
                          cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_and_save_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    https://stackoverflow.com/a/50386871/3215142
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.grid(False)

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig(file_name, bbox_inches='tight')
    plt.clf()


def plot_and_save_training_validation_plot(file_name, training_loss, validation_loss):
    # Use plot styling from seaborn.
    sns.set(style='darkgrid')

    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12, 6)

    # Plot the learning curve.
    plt.plot(training_loss, 'b-o', label="Training")
    plt.plot(validation_loss, 'g-o', label="Validation")

    # Label the plot.
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks([x+1 for x in range(len(training_loss))])
    plt.savefig(file_name, bbox_inches='tight')
    plt.clf()


def get_local_data_dir(experiment, tactic, mean_pooling_parameters, hidden_layer_used, num_of_hidden_units):
    return "data/experiment/{}/{}/{}/{}".format(experiment,
                                                tactic.replace(" ", "_"),
                                                get_mean_pooling_path_variable(
                                                    mean_pooling_parameters),
                                                get_hidden_layer_path_variable(
                                                    hidden_layer_used,
                                                    num_of_hidden_units))


def get_remote_data_dir(experiment, tactic, mean_pooling_parameters, hidden_layer_used, num_of_hidden_units):
    return '{}/{}/{}'.format(get_tactic_dir(experiment, tactic.replace(" ", "_")),
                                                              get_mean_pooling_path_variable(mean_pooling_parameters),
                                                              get_hidden_layer_path_variable(hidden_layer_used, num_of_hidden_units))


def get_mean_pooling_path_variable(mean_pooling_parameters):
    mean_pooling = "mean_pooling"

    mean_pooling_variable = "without" + "_" + mean_pooling
    mean_pooling_used = mean_pooling_parameters["use_mean_pooling"]
    max_sequences = mean_pooling_parameters["max_sequences"]
    if mean_pooling_used:
        mean_pooling_variable = "with" + "_" + mean_pooling
        mean_pooling_variable += "_" + str(max_sequences)

    return mean_pooling_variable


def get_hidden_layer_path_variable(hidden_layer_used, num_of_hidden_units):
    hidden_layer = "hidden_layer"

    hidden_layer_variable = "without" + "_" + hidden_layer

    if hidden_layer_used:
        hidden_layer_variable = "with" + "_" + hidden_layer \
                                + "_" + str(num_of_hidden_units)

    return hidden_layer_variable


def upload_results(tactic, experiment, mean_pooling_parameters, hidden_layer_used, num_of_hidden_units):
    LOCAL_DATA_DIR = get_local_data_dir(experiment, tactic, mean_pooling_parameters, hidden_layer_used, num_of_hidden_units)
    REMOTE_DATA_DIR = get_remote_data_dir(experiment, tactic, mean_pooling_parameters, hidden_layer_used, num_of_hidden_units)

    if not tf.io.gfile.exists(REMOTE_DATA_DIR):
        tf.io.gfile.makedirs(REMOTE_DATA_DIR)

    bucket = get_storage_bucket()

    #upload confusion matrix
    local_confusion_matrix_file = LOCAL_DATA_DIR + '/confusion_matrix.png'
    remote_confusion_matrix_file = REMOTE_DATA_DIR + '/confusion_matrix.png'
    upload_to_google_cloud(bucket, local_confusion_matrix_file, remote_confusion_matrix_file)

    #upload results
    local_results_file = LOCAL_DATA_DIR + '/results.csv'
    remote_results_file = REMOTE_DATA_DIR + "/results.csv"
    upload_to_google_cloud(bucket, local_results_file, remote_results_file)

    #compress folds data
    with ZipFile('folds_metrics.zip', 'w') as zipf:
        # writing each file one by one
        zipdir(get_local_data_dir(experiment, tactic, mean_pooling_parameters, hidden_layer_used, num_of_hidden_units) + "/", zipf)
    #upload compressed folds data
    local_compressed_file = 'folds_metrics.zip'
    remote_compressed_file = REMOTE_DATA_DIR + "/folds_metrics.zip"
    upload_to_google_cloud(bucket, local_compressed_file, remote_compressed_file)


def save_parameters(tactic, experiment, hyper_parameters, hidden_layer_parameters, mean_pooling_parameters):
    hidden_layer_used = hidden_layer_parameters['use_hidden_layer']
    num_of_hidden_units = hidden_layer_parameters['number_of_hidden_units']

    LOCAL_DATA_DIR = get_local_data_dir(experiment, tactic, mean_pooling_parameters,
                                        hidden_layer_used, num_of_hidden_units)
    REMOTE_DATA_DIR = get_remote_data_dir(experiment, tactic, mean_pooling_parameters,
                                          hidden_layer_used,
                                          num_of_hidden_units)
    BUCKET_PATH = get_bucket_path()

    if not tf.io.gfile.exists("{}/{}".format(BUCKET_PATH, REMOTE_DATA_DIR)):
        tf.io.gfile.makedirs("{}/{}".format(BUCKET_PATH, REMOTE_DATA_DIR))

    bucket = get_storage_bucket()

    parameters = dict()
    parameters['hyper_parameters'] = hyper_parameters
    parameters['hidden_layer_parameters'] = hidden_layer_parameters
    parameters['mean_pooling_parameters'] = mean_pooling_parameters

    with open("{}/parameters.json".format(LOCAL_DATA_DIR), 'w') as pj:
        json.dump(parameters, pj)

    local_parameters_file = LOCAL_DATA_DIR + '/parameters.json'
    remote_parameters_file = REMOTE_DATA_DIR + "/parameters.json"
    upload_to_google_cloud(bucket, local_parameters_file, remote_parameters_file)


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))


def upload_to_google_cloud(bucket, local_file, remote_file):
  blob2 = bucket.blob(remote_file)
  blob2.upload_from_filename(local_file)

def download_from_google_cloud(bucket, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

def create_report_experiment_1(mean_pooling_parameters, hidden_layer_used, num_of_hidden_units):
    # Opening JSON file
    file_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
    f = open("drive/My Drive/pytorch_notebooks/tactic_classifier/tactic_categories.json".format(file_directory))
    # returns JSON object as dict
    categories = json.load(f)
    experiment = "1"
    report_dir = get_report_dir(experiment, mean_pooling_parameters,
                                hidden_layer_used, num_of_hidden_units)

    report = pd.DataFrame([], columns=['Labels', 'Precision', 'Recall', 'F-Measure', 'sample_size'])
    for category in categories:
        for sub_category in category['sub_categories']:
            for tactic in sub_category['tactics']:
                '''
                class is the tactic
                '''
                tactic_name = tactic['class']
                REMOTE_DATA_DIR = get_remote_data_dir(experiment, tactic_name,
                                                      mean_pooling_parameters,
                                                      hidden_layer_used,
                                                      num_of_hidden_units)
                BUCKET_PATH = get_bucket_path()
                if not tactic['ignore']\
                        and tf.io.gfile.exists("{}/{}/results.csv".format(BUCKET_PATH,
                                                   REMOTE_DATA_DIR)):
                    # get number of samples for each class in tactic
                    samples = download_tactic_data(tactic_name, experiment)

                    distribution = collections.Counter(samples['class'])

                    # get results
                    results = pd.read_csv(
                        "{}/{}/results.csv".format(BUCKET_PATH,
                                                   REMOTE_DATA_DIR))
                    results = results.loc[results['Labels'] == tactic_name]
                    results = results[['Labels', 'Precision', 'Recall', 'F-Measure']]
                    results['sample_size'] = distribution[tactic_name]
                    report = report.append(results)

    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    #store report locally
    report = report.sort_values(by=['F-Measure'])
    report.round(2).to_csv(report_dir + "/report.csv", index = False)

    # upload report
    report_file_path = report_dir + "/report.csv"
    upload_to_google_cloud(get_storage_bucket(), report_file_path,
                           report_file_path)


def create_report_experiment_2(mean_pooling_parameters, hidden_layer_used, num_of_hidden_units):
    # Opening JSON file
    file_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
    f = open(
        "drive/My Drive/pytorch_notebooks/tactic_classifier/tactic_categories.json".format(
            file_directory))
    # returns JSON object as dict
    categories = json.load(f)
    experiment = "2"
    report_dir = get_report_dir(experiment, mean_pooling_parameters,
                                hidden_layer_used, num_of_hidden_units)

    report = pd.DataFrame([], columns=['Labels', 'Precision', 'Recall', 'F-Measure', 'sample_size'])

    for category in categories:
        for sub_category in category['sub_categories']:
            category_name = sub_category['name'].replace(" ", "_").replace("/", "_")
            if not sub_category['ignore']:
                # get number of samples for each class in tactic
                samples = download_tactic_data(category_name, experiment)
                distribution = collections.Counter(samples['class'])

                # get results
                REMOTE_DATA_DIR = get_remote_data_dir(experiment, category_name,
                                                      mean_pooling_parameters,
                                                      hidden_layer_used,
                                                      num_of_hidden_units)
                BUCKET_PATH = get_bucket_path()
                results = pd.read_csv("{}/{}/results.csv".format(BUCKET_PATH, REMOTE_DATA_DIR))
                results = results.loc[results['Labels'] == sub_category['name']]
                results = results[['Labels', 'Precision', 'Recall', 'F-Measure']]
                results['sample_size'] = distribution[sub_category['name']]
                report = report.append(results)

    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    #store report locally
    report = report.sort_values(by=['F-Measure'])
    report.round(2).to_csv(report_dir + "/report.csv", index = False)

    # upload report
    report_file_path = report_dir + "/report.csv"
    upload_to_google_cloud(get_storage_bucket(), report_file_path, report_file_path)


def create_report_experiment_3(mean_pooling_parameters, hidden_layer_used, num_of_hidden_units):
    # Opening JSON file
    file_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
    f = open(
        "drive/My Drive/pytorch_notebooks/tactic_classifier/tactic_categories.json".format(
            file_directory))
    # returns JSON object as dict
    categories = json.load(f)
    experiment = "3"
    report_dir = get_report_dir(experiment, mean_pooling_parameters, hidden_layer_used, num_of_hidden_units)

    for category in categories:

        for sub_category in category['sub_categories']:
            report = pd.DataFrame([], columns=['Labels', 'Precision', 'Recall',
                                               'F-Measure', 'sample_size'])
            category_name = sub_category['name'].replace(" ", "_").replace("/", "_")

            if not sub_category['ignore']:
                # get number of samples for each class in tactic
                samples = download_tactic_data(category_name, experiment)
                distribution = collections.Counter(samples['class'])

                # get results
                REMOTE_DATA_DIR = get_remote_data_dir(experiment, category_name,
                                                      mean_pooling_parameters,
                                                      hidden_layer_used,
                                                      num_of_hidden_units)
                BUCKET_PATH = get_bucket_path()
                results = pd.read_csv("{}/{}/results.csv".format(BUCKET_PATH, REMOTE_DATA_DIR))
                results = results[['Labels', 'Precision', 'Recall', 'F-Measure']]

                results['sample_size'] = results['Labels'].apply(lambda x : distribution[x])
                report = report.append(results)

                if not os.path.exists(report_dir):
                    os.makedirs(report_dir)
                # store report locally
                report = report.sort_values(by=['F-Measure'])
                report.round(2).to_csv(report_dir + "/" + category_name + ".csv", index=False)

                # upload report
                report_file_path = report_dir + "/" + category_name + ".csv"
                upload_to_google_cloud(get_storage_bucket(), report_file_path,
                                       report_file_path)

def create_report_experiment_5(mean_pooling_parameters, hidden_layer_used, num_of_hidden_units):
    # Opening JSON file
    file_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
    f = open(
        "drive/My Drive/pytorch_notebooks/tactic_classifier/tactic_categories.json".format(
            file_directory))
    # returns JSON object as dict
    categories = json.load(f)
    experiment = "5"
    report_dir = get_report_dir(experiment, mean_pooling_parameters, hidden_layer_used, num_of_hidden_units)
    tactic_name = "tactic"
    samples = download_tactic_data(tactic_name, experiment)
    distribution = collections.Counter(samples['class'])

    # get results
    REMOTE_DATA_DIR = get_remote_data_dir(experiment, tactic_name,
                                          mean_pooling_parameters,
                                          hidden_layer_used,
                                          num_of_hidden_units)
    BUCKET_PATH = get_bucket_path()
    results = pd.read_csv("{}/{}/results.csv".format(BUCKET_PATH, REMOTE_DATA_DIR))
    results = results[['Labels', 'Precision', 'Recall', 'F-Measure']]

    results['sample_size'] = results['Labels'].apply(lambda x: distribution[x])
    report = pd.DataFrame([], columns=['Labels', 'Precision', 'Recall',
                                       'F-Measure', 'sample_size'])
    report = report.append(results)

    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    # store report locally
    report = report.sort_values(by=['F-Measure'])
    report.round(2).to_csv(report_dir + "/" + tactic_name + ".csv",index=False)

    # upload report
    report_file_path = report_dir + "/" + tactic_name + ".csv"
    upload_to_google_cloud(get_storage_bucket(), report_file_path,
                           report_file_path)


def create_report_experiment_6(mean_pooling_parameters, hidden_layer_used, num_of_hidden_units):
    # Opening JSON file
    file_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
    f = open(
        "drive/My Drive/pytorch_notebooks/tactic_classifier/tactic_categories.json".format(
            file_directory))
    # returns JSON object as dict
    categories = json.load(f)
    experiment = "6"
    report_dir = get_report_dir(experiment, mean_pooling_parameters, hidden_layer_used, num_of_hidden_units)

    for category in categories:

        for sub_category in category['sub_categories']:
            report = pd.DataFrame([], columns=['Labels', 'Precision', 'Recall',
                                               'F-Measure', 'sample_size'])

            if not sub_category['ignore']:
                category_name = sub_category['file_name'].replace(" ", "_").replace(
                    "/", "_")
                # get number of samples for each class in tactic
                samples = download_tactic_data(category_name, experiment)
                distribution = collections.Counter(samples['class'])

                # get results
                REMOTE_DATA_DIR = get_remote_data_dir(experiment, category_name,
                                                      mean_pooling_parameters,
                                                      hidden_layer_used,
                                                      num_of_hidden_units)
                BUCKET_PATH = get_bucket_path()
                results = pd.read_csv("{}/{}/results.csv".format(BUCKET_PATH, REMOTE_DATA_DIR))
                results = results[['Labels', 'Precision', 'Recall', 'F-Measure']]

                results['sample_size'] = results['Labels'].apply(lambda x : distribution[x])
                report = report.append(results)

                if not os.path.exists(report_dir):
                    os.makedirs(report_dir)
                # store report locally
                report = report.sort_values(by=['F-Measure'])
                report.round(2).to_csv(report_dir + "/" + category_name + ".csv", index=False)

                # upload report
                report_file_path = report_dir + "/" + category_name + ".csv"
                upload_to_google_cloud(get_storage_bucket(), report_file_path,
                                       report_file_path)

def get_files_with_prefix(bucket_name, prefix, delimiter=None):
    """Lists all the blobs in the bucket that begin with the prefix.

    This can be used to list all blobs in a "folder", e.g. "public/".

    The delimiter argument can be used to restrict the results to only the
    "files" in the given "folder". Without the delimiter, the entire tree under
    the prefix is returned. For example, given these blobs:

        a/1.txt
        a/b/2.txt

    If you just specify prefix = 'a', you'll get back:

        a/1.txt
        a/b/2.txt

    However, if you specify prefix='a' and delimiter='/', you'll get back:

        a/1.txt

    Additionally, the same request will return blobs.prefixes populated with:

        a/b/
    """

    storage_client = storage.Client(project=get_storage_client_project())

    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = storage_client.list_blobs(
        bucket_name, prefix=prefix, delimiter=delimiter
    )

    files = []
    for blob in blobs:
        files.append(blob.name)

    return files

def download_reports(experiment):
    report_base_dir = '{}/reports'.format(get_experiment_dir(experiment))
    reports = get_files_with_prefix(get_bucket_name(), report_base_dir)

    for report in reports:
        report_dir = os.path.dirname(report)
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
        download_from_google_cloud(get_storage_bucket(), report, report)

