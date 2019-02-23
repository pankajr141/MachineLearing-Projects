# Sequence labelling

### Description
Sequence labelling models are used to label sequences, they will tag each token/word of the sequence with a label, which will be later used for extracting information.  

**Example** - Suppose we are making a sequence model to extract city names so out tag file consist of 2 tokens CITY and O meaning other.  
<sub>
**Sentence** 		- I am living in Mumbai  
**Sequence Labels** - O  O   O 	   O CITY  	
</sub>

### Directory Structure
<pre>
├── application.py            - contains all the dataset preparation and model calling code
├── create_training_data.py   - contains code for preparing training data.
├── dataset_pubmed
│   ├── pubmed_dataset.csv
│   ├── test.csv
│   └── train.csv
├── helper.py                 - helper functions
├── labels.txt
├── model.py                  - actual model code
├── print_result.py           - contains code for analyzing result files by plotting color based output
├── sentences.txt
└── tags.txt
</pre>
### Architecture

## Model Execution Details
### Dataset Creation
### Training the model
<pre>
python application.py --mode TRAIN --datafile "pubmed_train_x|pubmed_train_y,pubmed_test_x|pubmed_test_y"
</pre>
As the name suggest we are passing train and test file in --datafile argument. x is sentence file and y is label file. Please see data folder for their structure.

### Evaluting the model
<pre>
python application.py --mode EVAL --datafile "pubmed_train_x|pubmed_train_y"
</pre>

If you want to use a different model for evaluation
<pre>
python application.py --mode EVAL --datafile "pubmed_train_x|pubmed_train_y" --modeldir models_step=1400_loss=0.38
</pre>

### Predictions
Here pass the --resultfile argument to save output in that file.
<pre>
python application.py --mode PREDICT --datafile "pubmed_train_x_lower" --resultfile pubmed_train_result
</pre>

If you want to use a different model for prediction
<pre>
python application.py --mode PREDICT --datafile "pubmed_train_x_lower" --resultfile pubmed_train_result --modeldir models_step=1400_loss=0.38
</pre>

<sub><b><i>
Note: Remember since this is prediction we have kept batch size as 1, so that we dont add unnecessary pads to sentence, this make the predictions to run slow.
Batch Size can be changed in codebase, but since we are not maintaining any ID to identify the sentence and relying on line number only, please take care of that part.
</i></b></sub>

### Analyze result
In order to analyze result pass the result file generated during prediction as below

To print all the sentences on screen
python -W ignore print_result.py --datafile pubmed_test_x.csv --resultfile result_test  --tagfile tags.txt

To print single sentence by index on screen
python -W ignore print_result.py --datafile pubmed_test_x.csv --resultfile result_test  --tagfile tags.txt --index 1

<sub><i>
  Note: Above output is color coded, by default it support 3 Tags, colors being  
  **RED:**  <tab>FN (missed labels)  
  **GREEN, PURPLE:** TP (Correctly predicted for class 1 and 2)  
  **UNDERLINE:** FP (wrong preduction)  
</i></sub>

