import BertMeanPooling as bmp
import utils
from transformers import AdamW, BertTokenizer

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import TensorDataset
import numpy as np
import random
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import time
import datetime
from transformers import get_linear_schedule_with_warmup
import matplotlib.pyplot as plt

import os

class TacticClassifier:

    def __init__(self, tactic, experiment):
        self.tactic = tactic
        self.experiment = experiment

    def use_gpu(self):
        '''
        Set torch to use gpu
        :return:
        '''
        import torch
        # Tell PyTorch to use the GPU.
        self.device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))

    def initialize_model(self, bert_model='bert-base-uncased'):
        self.model = bmp.BertMeanPooling(bert_model)
        utils.to(self.model, self.device)

    def initialize_tokenizer(self, bert_model='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(bert_model,
                                                       do_lower_case=True)

    def initialize_classifier(self, num_labels, use_hidden_layers=False,
                              num_of_hidden_units = 128):
        self.model.initialize_classifier(self.device,
                                         use_hidden_layers,
                                         num_of_hidden_units,
                                         num_labels)

    def initialize_hyperparameters(self, hyper_parameters, train_data_size):
        # Number of training epochs. The BERT authors recommend between 2 and 4.
        # We chose to run for 4, but we'll see later that this may be over-fitting the
        # training data.
        self.epochs = hyper_parameters['epochs']
        # The DataLoader needs to know our batch size for training, so we specify it
        # here. For fine-tuning BERT on a specific task, the authors recommend a batch
        # size of 16 or 32.
        self.batch_size = hyper_parameters['batch_size']
        self.learning_rate = hyper_parameters['learning_rate']
        self.optimizer = AdamW(self.model.parameters(),
                               lr = self.learning_rate,
                               # args.learning_rate - default is 5e-5, our notebook had 2e-5
                               eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                               )

        # Total number of training steps is [number of batches] x [number of epochs].
        # (Note that this is not the same as the number of training samples).
        total_steps = train_data_size * self.epochs

        # Create the learning rate scheduler.
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=0,
                                                         # Default value in run_glue.py
                                                         num_training_steps=total_steps)

    def seed(self):
        # Set the seed value all over the place to make this reproducible.
        seed_val = 42

        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

    def run_experiment(self, hyper_parameters, hidden_layer_parameters, mean_pooling_parameters):
        '''
        :param use_hidden_layer: whether to use hidden layer in the classifier
        :param number_of_hidden_units: number of units in the hidden layer
        :param use_mean_pooling: whether to use mean pooling or not.
        If mean pooling is not used, first max_seq_length tokens of the input
        are considered
        :return:
        '''
        use_mean_pooling = mean_pooling_parameters['use_mean_pooling']
        max_sequences = mean_pooling_parameters['max_sequences']
        use_hidden_layer = hidden_layer_parameters['use_hidden_layer']
        number_of_hidden_units = hidden_layer_parameters['number_of_hidden_units']

        LOCAL_BASE_DIR = utils.get_local_data_dir(self.experiment,
                                                  self.tactic,
                                                  mean_pooling_parameters,
                                                  use_hidden_layer,
                                                  number_of_hidden_units)
        if not os.path.exists(LOCAL_BASE_DIR):
            os.makedirs(LOCAL_BASE_DIR)

        utils.save_parameters(self.tactic, self.experiment, hyper_parameters,
                              hidden_layer_parameters, mean_pooling_parameters)

        data = utils.download_tactic_data(self.tactic, self.experiment)
        text = data['text']
        labels = data['class']
        print_cuda_stats('starting experiment')
        self.use_gpu()
        self.seed()
        self.initialize_tokenizer()

        '''
        When using mean pooling approach, you want to use the entire input text. 
        When not using mean pooling, you can't use the entire text because of 
        maximum sequence length constraint. 
        We use the leading max_seq_length number of tokens when not using mean pooling.
        '''
        use_leading_tokens = not use_mean_pooling
        tokenized_text, no_of_parts = utils.tokenize_input_text(text, self.tokenizer, use_leading_tokens=use_leading_tokens)

        # do a 10 fold cross validation
        kfold = StratifiedKFold(10, True, 1)

        loss_function = torch.nn.CrossEntropyLoss()

        split_no = 1  # used to create folders for each split
        le = utils.get_label_encoder(labels)

        unique_labels = list(labels.unique())
        unique_labels.sort()

        labels = le.transform(labels)
        labels = torch.tensor(labels)

        folds_confusion_matrix = []
        folds_precision_recall_f1score_support = []
        folds_validation_accuracy = []

        for train, validation in kfold.split(tokenized_text, labels):
            FOLD_DIR = "{}/{}".format(LOCAL_BASE_DIR, split_no)
            if not os.path.exists(FOLD_DIR):
                os.makedirs(FOLD_DIR)

            print_cuda_stats('initializing model')
            self.initialize_model()
            print_cuda_stats('model initialized')
            self.initialize_classifier(len(unique_labels),
                                       use_hidden_layers = use_hidden_layer,
                                       num_of_hidden_units = number_of_hidden_units)

            train_dataset = TensorDataset(tokenized_text[train], labels[train])
            val_dataset = TensorDataset(tokenized_text[validation],
                                        labels[validation])
            print('{:>5,} training samples'.format(len(train_dataset)))
            print('{:>5,} validation samples'.format(len(val_dataset)))

            self.initialize_hyperparameters(hyper_parameters, len(train_dataset))

            # Create the DataLoaders for our training and validation sets.
            # We'll take training samples in random order.
            train_dataloader = utils.create_data_loader(train_dataset,
                                                        RandomSampler,
                                                        self.batch_size)

            # For validation the order doesn't matter, so we'll just read them sequentially.
            validation_dataloader = utils.create_data_loader(val_dataset,
                                                             SequentialSampler,
                                                             self.batch_size)

            # We'll store a number of quantities such as training and validation loss,
            # validation accuracy, and timings.
            training_stats = []

            # Measure the total training time for the whole run.
            total_t0 = time.time()

            best_f1_score = None
            best_loss = None
            best_metrics = None

            # For each epoch...
            for epoch_i in range(0, self.epochs):

                avg_train_loss, training_time = self.train(train_dataloader, epoch_i,
                                            loss_function, max_sequences)

                print("")
                print("  Average training loss: {0:.2f}".format(avg_train_loss))
                print("  Training epoch took: {:}".format(training_time))

                t0 = time.time()
                val_metrics = self.validate(validation_dataloader, loss_function, le, max_sequences)
                # Measure how long the validation run took.
                validation_time = utils.format_time(time.time() - t0)

                print("  Validation Loss: {0:.2f}".format(val_metrics["avg_val_loss"]))
                print("  Validation took: {:}".format(validation_time))

                # Record all statistics from this epoch.
                training_stats.append(
                    {
                        'epoch': epoch_i + 1,
                        'Training Loss': avg_train_loss,
                        'Valid. Loss': val_metrics["avg_val_loss"],
                        'Valid. Accur.': val_metrics["avg_val_accuracy"],
                        'Training Time': training_time,
                        'Validation Time': validation_time
                    }
                )

                if best_f1_score == None or \
                        (val_metrics["precision_recall_fscore_support"][2].mean() >= best_f1_score and val_metrics["avg_val_loss"] < best_loss):
                    best_metrics = val_metrics
                    best_f1_score = best_metrics["precision_recall_fscore_support"][2].mean()
                    best_loss = best_metrics["avg_val_loss"]

            #epoch with best f1-score and lowest loss
            folds_confusion_matrix.append(best_metrics['confusion_matrix'])
            folds_precision_recall_f1score_support.append(best_metrics['precision_recall_fscore_support'])
            folds_validation_accuracy.append(best_metrics['avg_val_accuracy'])

            #save fold metrics
            fold_results = []
            # save average metrics as results
            # score includes precision, recall, f-measure and support
            for i, label in enumerate(unique_labels):
                fold_results.append(
                    [label.replace("_", " "),
                     best_metrics['precision_recall_fscore_support'][0][i],
                     best_metrics['precision_recall_fscore_support'][1][i],
                     best_metrics['precision_recall_fscore_support'][2][i],
                     best_metrics['precision_recall_fscore_support'][3][i]])
            fold_results_df = pd.DataFrame(fold_results,
                              columns=['Labels', 'Precision', 'Recall',
                                       'F-Measure',
                                       'Support'])
            fold_results_df.to_csv("{}/results.csv".format(FOLD_DIR), index=False, header=True)

            print("")
            print("Training complete for split-{}!".format(str(split_no)))
            print("Total training took {:} (h:mm:ss)".format(utils.format_time(time.time() - total_t0)))


            # Display floats with two decimal places.
            pd.set_option('precision', 2)

            # Create a DataFrame from our training statistics.
            df_stats = pd.DataFrame(data=training_stats)

            # Use the 'epoch' as the row index.
            df_stats = df_stats.set_index('epoch')

            # Display the table.
            print(df_stats)

            # save metrics for fold
            df_stats.to_csv("{}/training_validation_stats.csv".format(FOLD_DIR))

            utils.plot_and_save_training_validation_plot("{}/train_vs_valiation_loss_plot.png".format(FOLD_DIR),
                                                         df_stats[
                                                             'Training Loss'],
                                                         df_stats[
                                                             'Valid. Loss'])
            split_no += 1

        #average fold metrics
        avg_confusion_matrix = np.mean(folds_confusion_matrix, axis=0)
        avg_precision_recall_f1score_support = np.mean(folds_precision_recall_f1score_support, axis=0)

        utils.plot_and_save_confusion_matrix(LOCAL_BASE_DIR + "/confusion_matrix.png",
                                             avg_confusion_matrix, unique_labels,
                                       cmap=plt.cm.Blues,
                                       normalize=False)
        results = []
        #save average metrics as results
        # score includes precision, recall, f-measure and support
        for i, label in enumerate(unique_labels):
            results.append(
                [label.replace("_", " "), avg_precision_recall_f1score_support[0][i], avg_precision_recall_f1score_support[1][i],
                 avg_precision_recall_f1score_support[2][i], avg_precision_recall_f1score_support[3][i]])
        df = pd.DataFrame(results,
                          columns=['Labels', 'Precision', 'Recall', 'F-Measure',
                                   'Support'])
        df.to_csv(LOCAL_BASE_DIR + '/results.csv', index=False,
                  header=True)

        # upload confusion matrix
        # upload metrics
        # upload each fold metrics
        utils.upload_results(self.tactic,
                             self.experiment,
                             mean_pooling_parameters,
                             use_hidden_layer,
                             number_of_hidden_units)


    def train(self, train_dataloader, epoch, loss_function, max_sequences):
        # ========================================
        #               Training
        # ========================================

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Perform one full pass over the training set.
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1,
                                                         self.epochs))
        print('Training...')

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the model into training mode. Don't be mislead--the call to
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        self.model.train()

        print_cuda_stats('start training')

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = utils.format_time(time.time() - t0)

                # Report progress.
                print(
                    '  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(
                        step, len(train_dataloader), elapsed))

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because
            # accumulating the gradients is "convenient while training RNNs".
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            self.model.zero_grad()
            print_cuda_stats('zero grad')
            # Perform a forward pass (evaluate the model on this training batch).
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # It returns different numbers of parameters depending on what arguments
            # arge given and what flags are set. For our useage here, it returns
            # the loss (because we provided labels) and the "logits"--the model
            # outputs prior to activation.
            document_batch = batch[0]
            batch_predictions = self.model(document_batch,
                                           max_sequences,
                                           self.device)
            print_cuda_stats('forward')
            # batch_predictions = utils.to(batch_predictions, self.device)
            batch_correct_output = utils.to(batch[1], self.device)
            loss = loss_function(batch_predictions,
                                 batch_correct_output)
            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_train_loss += loss.item()
            print_cuda_stats('accumulate loss')

            # Perform a backward pass to calculate the gradients.
            loss.backward()
            print_cuda_stats('backward')

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            self.optimizer.step()
            print_cuda_stats('optimizer step')
            # Update the learning rate.
            self.scheduler.step()
            print_cuda_stats('delete loss')

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = utils.format_time(time.time() - t0)

        return avg_train_loss, training_time

    def validate(self, validation_dataloader, loss_function, le, max_sequences):
        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print("Running Validation...")

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        self.model.eval()

        # Tracking variables
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        all_predictions = []
        all_labels = []

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                document_batch = utils.to(batch[0], self.device)
                batch_predictions = self.model(document_batch,
                                               max_sequences,
                                          self.device)
                batch_correct_output = utils.to(batch[1],self.device)
                batch_predictions = utils.to(batch_predictions, self.device)
                loss = loss_function(batch_predictions,
                                     batch_correct_output)

            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            logits = batch_predictions.detach().cpu().numpy()
            label_ids = batch_correct_output.to('cpu').numpy()
            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_accuracy += utils.flat_accuracy(logits, label_ids)

            logits = logits.argmax(axis=1)
            all_predictions.extend(list(logits))
            all_labels.extend(label_ids)

        all_labels = list(le.inverse_transform(all_labels))
        all_predictions = list(le.inverse_transform(all_predictions))
        print(classification_report(all_labels, all_predictions, output_dict=False))
        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(
            validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)
        metrics = dict({
            "avg_val_loss": avg_val_loss,
            "avg_val_accuracy": avg_val_accuracy,
            "precision_recall_fscore_support":precision_recall_fscore_support(all_labels, all_predictions,
                                            average=None),
            "confusion_matrix":confusion_matrix(all_labels, all_predictions)
        })

        return metrics


show_stats = False


def print_cuda_stats(msg):
    if show_stats:
        print('--', msg)
        print('allocated: %dM, max allocated: %dM, cached: %dM, max cached: %dM' % (
            torch.cuda.memory_allocated() / 1024 / 1024,
            torch.cuda.max_memory_allocated() / 1024 / 1024,
            torch.cuda.memory_cached() / 1024 / 1024,
            torch.cuda.max_memory_cached() / 1024 / 1024
        ))