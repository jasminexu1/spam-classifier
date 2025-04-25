# Distributed Spam Classifier

This project implements a distributed spam classifier in Apache Spark using stochastic gradient descent (SGD).  
It replicates and extends the techniques from the paper *Efficient and Effective Spam Filtering and Re-ranking for Large Web Datasets* by Cormack, Smucker, and Clarke.

The classifier is trained on hashed byte 4-gram features and supports:
- Training and prediction on large-scale datasets
- Evaluation using ROC-based metrics
- Ensemble methods (score averaging and majority voting)
- Optional data shuffling to explore the effect of SGD input order

