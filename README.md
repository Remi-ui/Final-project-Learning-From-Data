# Final-project-Learning-From-Data
Final project for Learning From Data


<h2>Installation of dependencies</h2>

The required dependencies can be installed with:
```
pip install -r requirements.txt
```

<h2>How to train the model</h2>

5 different models can be trained: Naive Bayes, an SVM, an optimized SVM, and LSTM and a BERT model. These can all be specified in the command line by using --model <model> while running evaluate.py. By default this will be evaluated on "newspapers_157_upsampled_test.json" (no command line argument is needed when evaluating on the test set), however a different file can be specified with --eval path/to/file. If you would like to evaluate an SVM on the development set for example use the following command:

```
python3 evaluate.py --model svm --eval newspapers_157_upsampled_dev.json
```
--model accepts: naive_bayes, svm, svm_optimized, lstm, bert. By default it will train an SVM.
<h2>How to preprocess the data so it is in the correct format</h2>  

<h2>How to train the models on the data</h2>

<h2>How to use one of your already trained models to predict on unseen data</h2>

<h2>How to evaluate an output file using the gold standard</h2>

Run output_evaluate.py and specify the path to the desired prediction_vs_gold file. If (for example) the desired file is located in experiments/naive_bayes, run the following command:
```
python3 output_evaluate.py --path experiments/naive_bayes/prediction_vs_gold.txt
```
