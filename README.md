# population-unveiler
Bagging OCSVM for unveiling the full stellar population of stellar groups.

Using a predefined training set and a sample of potential member stars, i.e. the prediction set, the script searches for viable OCSMV classifiers in a predefined hyper-parameter range. Classifiers which are judged to fulfill our prior assumtions about the distribution of sources in 5D are admitted to the classifier ensemble. The ensemble then infers the membership of unseen points and saves them in a file alongside the hyper-parameters and accepted models. For more details on the application of this method to the Meingast 1 stream where we uncovered about a factor 10 more sources see our [paper](https://arxiv.org/abs/2002.05728).

To execute the script (with some default values) call the following
```
python3 hyperparam_search.py --n_cores=8 --n_searches=1500 --cp_lo=0.1 --cp_hi=10 --fpath_train=/path/to/training_data.fits --fpath_predict=/path/to/prediction_data.fits
```
