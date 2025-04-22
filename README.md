1. Build dependencies: pip install -r /path/to/requirements.txt
2. Preprocess the datasets and save them by running data_preprocess.py. Make sure to change the path names to save the processed datasets.
3. Navigate to Training folder. We have four files for the 4 models: Baseline, unidirectional LSTM, Bidirectional LSTM, Bidirectional LSTM with max pooling. Save the checkpoints in dedicated fodlers. Make sure to change the path names.
4.  Navigate to Validation folder. We have four files for the 4 models: Baseline, unidirectional LSTM, Bidirectional LSTM, Bidirectional LSTM with max pooling. Run them to get the performance of these models in VAL split of SNLI.
5.  Navigate to Testing folder. We have four files for the 4 models: Baseline, unidirectional LSTM, Bidirectional LSTM, Bidirectional LSTM with max pooling. Run them to get the performance of these models in TEST split of SNLI.
6.  Navigate to SentEval folder. We have four files for the 4 models: Baseline, unidirectional LSTM, Bidirectional LSTM, Bidirectional LSTM with max pooling. Run them to get the performance of the sentence embeddings on a diverse set of downstream tasks called ‘transfer’ tasks.


Link to download the checkpoints and other pre-processed datasets: 
https://amsuni-my.sharepoint.com/:f:/g/personal/soham_chatterjee2_student_uva_nl/EkhJwKEsXzRAmiZKSSb198QBrBCD9kksHXZfN8V49_Emeg?e=fAtGBP
