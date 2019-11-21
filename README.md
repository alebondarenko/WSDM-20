# Comparative Web Search Questions

	@InProceedings{stein:2020a,
	  author =              {Alexander Bondarenko and Pavel Braslavski and Michael V{\"o}lske and Rami Aly and Maik Fr{\"o}be and Alexander Panchenko and Chris Biemann and Benno Stein and Matthias Hagen},
	  booktitle =           {13th ACM International Conference on Web Search and Data Mining (WSDM 2020)},
	  month =               feb,
	  publisher =           {ACM},
	  site =                {Houston, USA},
	  title =               {{Comparative Web Search Questions}},
	  year =                2020
	}

[[Paper Link](https://webis.de/downloads/publications/papers/stein_2020a.pdf)]

## Code structure
### Notebooks:
- rule_based_classification.ipynb to perform a rule-based classification (15 patterns in Russian)
- question_binary_classification.ipynb to perform a rule-based classification
- multiclass_classification.ipynb to perform a rule-based classification

The two last notebooks will use predictions produced by CNN and BERT as decision probabilities.

### Classification with Neural Networks

Repository for neural models implemented for the classification of comparative questions in a binary and multi-label setting.

#### System Requirement

The system was tested on Debian/Ubuntu Linux with a GTX 1080TI and TITAN X.

General Settings and Parameters for models:

| Option |  Description | Default|
|--------|-------------|---|
| --sequence_length | Maximum sequence input length of text | 100 |
| --epochs | Number of epochs to train the classifier | 60 |
| --use_statc | Whether the embedding layer should not be trainable | False |
| --use_early_stop |Uses early stopping during training | False |
| --batch_size |Set minibatch size | 32 |
| --learning_rate |The learning rate of the classifier | 0.0005 |
| --learning_decay |Whether to use learning decay, 1 indicates no decay, 0 max.| 1 |
| --init_layer |Whether to initialize the final layer with label co-occurence.| False |
| --iterations |How many classifiers to be trained, only relevant for train_n_models_final | 3 |
| --activation_th |Activation threshold of the final layer | 0.5 |
| --adjust_hierarchy |Postprocessing hierarchy correction | None|
| --correction_th |Threshold for threshold-label correction method | False |

Please note, that `--init_layer, --correction_th --adjust_hierarchy` are only usable, if the hierarchy of a dataset is given as input as well.


1. Clone this repository: https://github.com/huggingface/pytorch-pretrained-BERT
2. Install requirements.
2. Replace the run_classifier.py with the contents in run_bert.py.
3. Run the command:
*Example for testing:*
python3 run_classifier.py   --task_name COMPQ --do_eval --do_lower_case --data_dir path_to_input_data --bert_model your_bert_model  
--max_seq_length 128 --train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir path_to_model --load_finetuned_model

With:
--data_dir The path to the input data. Please name the input data test-binary.tsv, or change the filename in run_classifier.py.
--output_dir this is where you have to put in the model and config file.

After running, the output should also already be created in the output folder.


CNN settings:

| Option |  Description | Default|
|--------|-------------|---|
| --num_filters | Number of filters for each window size | 500 |

*Example for training model:*  
python3 main.py --mode train --classifier cnn --use_cv --use_early_stop --num_filters 100 
--learning_rate 0.0001 --lang COMPQ_BINARY_YAN --sequence_length 15`

*Example for testing model:* 
python3 run_cnn.py --input_path path_to_test_data --cnn_path /checkpoints/your_cnn_model.h5 
--vocabulary_path your_vocabulary_cnn --threshold 0.5

## Data
Annotated English question queries and pre-trained models are available from: [[Data Link](https://xxx)] 