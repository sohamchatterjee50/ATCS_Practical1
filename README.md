1. Build dependencies: pip install -r /path/to/requirements.txt
2. Preprocess the datasets and save them by running data_preprocess.py. Make sure to change the path names to save the processed datasets.
3. Navigate to Training folder. We have four files for the 4 models: Baseline, unidirectional LSTM, Bidirectional LSTM, Bidirectional LSTM with max pooling. Save the checkpoints in dedicated fodlers. Make sure to change the path names.
4.  Navigate to Validation folder. We have four files for the 4 models: Baseline, unidirectional LSTM, Bidirectional LSTM, Bidirectional LSTM with max pooling. Run them to get the performance of thse models in VAL split of SNLI.
5.  Navigate to Testing folder. We have four files for the 4 models: Baseline, unidirectional LSTM, Bidirectional LSTM, Bidirectional LSTM with max pooling. Run them to get the performance of thse models in TEST split of SNLI.
6.  
