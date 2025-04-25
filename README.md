# Distributed Spam Classifier

This project implements a distributed spam classifier in Apache Spark using stochastic gradient descent (SGD).  
It replicates and extends the techniques from the paper *Efficient and Effective Spam Filtering and Re-ranking for Large Web Datasets* by Cormack, Smucker, and Clarke.

The classifier is trained on hashed byte 4-gram features and supports:
- Training and prediction on large-scale datasets
- Evaluation using ROC-based metrics
- Ensemble methods (score averaging and majority voting)
- Optional data shuffling to explore the effect of SGD input order

(Originally assignment for distributed computing class, so the data and c programs in the following commands cannot be posted publicly, please contact me to get those data)

Steps to run:
1. Compile the C program: ```gcc -O2 -o compute_spam_metrics compute_spam_metrics.c -lm```
2. Compile the program: ```mvn clean package```
3. Run the trainer: 
```
spark-submit --driver-memory 2g --class SpamTrainer target/spam-classifier.jar --input spam.train.group_x.txt --model output-model-group_x
```
4. Run the predictor: 
```
spark-submit --driver-memory 2g --class SpamPredictor target/spam-classifier.jar --input spam.test.qrels.txt --output test-group_x --model output-model-group_x
```
5. Evaluate: ```./spam_eval.sh test-group_x```

For the EnsembleSpamPredictor:
1. Run 
```
mkdir model-fusion
cp model-group_x/part-00000 model-fusion/part-00000
cp model-group_y/part-00000 model-fusion/part-00001
cp model-britney/part-00000 model-fusion/part-00002
```
2. Run 
```
spark-submit --driver-memory 2g --class EnsembleSpamPredictor target/spam-classifier.jar --input spam.test.qrels.txt --output test-fusion-average --model model-fusion --method average
```
to use the ensemble predictor to combine model outputs by averaging or voting.