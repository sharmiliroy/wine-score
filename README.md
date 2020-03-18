# wine-score

This repository contains the scripts for training, deploying and evaluation a wine-score model in AWS SageMaker using a custom PyTorch model. 

## who uses wine-score

People like me who have no idea how to buy a good wine and wonder helplessly at the descriptions on the bottle. This model can be a wine-assistant for such novices. 

## how it works

1. wine-score is based on a bidirectional LSTM architecture. It has been trained to predict the wine score given a string that contains the wine-type, place of manufacture (region, province, country, vineyar), price and other available information.

2. The model has been trained using 150k wine descriptions and ratings downloaded from Kaggle. The link for the data can be found here https://www.kaggle.com/zynicide/wine-reviews

3. The model outputs 'average', 'good', 'excellent' label given a string that contains information about the wine such as type, place of origin, flavors, price etc.

## pre-requisites
1. AWS SageMaker
2. Pytorch
3. Pandas
4. Numpy
5. NLTK
6. scikit-learn
7. matplotlib

## data
The wine-reviews used to train the data is downloaded from the Kaggle dataset here https://www.kaggle.com/zynicide/wine-reviews

## results
The current accuracy of prediction for the model is 69.5%.
