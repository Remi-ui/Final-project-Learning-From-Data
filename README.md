# Final-project-Learning-From-Data
This file contains information about how to run, experiment with and evaluate the different models for the final project for Learning from Data. For questions or concerns please reach out to:

Remi Thüss - r.m.thuss@student.rug.nl

Robert van Timmeren - r.j.van.timmeren@student.rug.nl

<h2>Installation of dependencies</h2>

The required dependencies can be installed with:
```
pip install -r requirements.txt
```

<h2>Additional dependencies</h2>

The LSTM model makes use of pre-trained static embeddings (GloVe). Due to upload constraints of Github you should supply your own .txt embeddings in the glove folder and name them glove.txt. This means that the program is able to locate your embeddings at: ../Final-project-Learning-From-Data/glove/glove.txt.

You can download the GloVe embeddings at: https://nlp.stanford.edu/projects/glove/. For our research, we made use of the Wikipedia 2014 + Gigaword 5 100d variant.

<h2>How to train the model</h2>

5 different models can be trained: Naive Bayes, an SVM, an optimized SVM, and LSTM (GloVe) and a BERT model. These can all be specified in the command line by using --model <model> while running evaluate.py. By default this will be evaluated on "newspapers_157_upsampled_test.json" (no command line argument is needed when evaluating on the test set), however a different file can be specified with --eval path/to/file. If you would like to evaluate an SVM on the development set for example use the following command:

```
python3 evaluate.py --model svm --eval newspapers_157_upsampled_dev.json
```
--model accepts: naive_bayes, svm, svm_optimized, lstm, bert. By default it will train an SVM.
<h2>How to preprocess the data so it is in the correct format</h2>
If your file is in the format of other COP meeting files, no preprocessing is needed. If for example the file you wish to evaluate a model on is called "COP25.filt3.sub.json". You can evaluate a model using this file as follows:

```
python3 evaluate.py --model svm --eval COP25.filt3.sub.json
```

<h2>How to use one of your already trained models to predict on unseen data</h2>

 As both BERT models take quite some time to train some models are available to use straight away. These models can be found at: INSERT_LINK.com. If you'd like to run model1.h5 for example. Use the following command
 ```
 python3 evaluate.py --model bert --saved_model model1.h5
 ```
 
<h2>How to evaluate an output file using the gold standard</h2>

Run output_evaluate.py and specify the path to the desired prediction_vs_gold file (by default output files are output to experiments/prediction_vs_gold.txt). If (for example) the desired file is located in experiments/naive_bayes, run the following command:
```
python3 output_evaluate.py --path experiments/naive_bayes/prediction_vs_gold.txt
```
The above code will generate a classification report with scores for each class.
If you'd like to evaluate a file that was generated from a bert model --bert needs to be set to true as follows:
```
python3 output_evaluate.py --path experiments/path/prediction_vs_gold.txt --bert True
```
