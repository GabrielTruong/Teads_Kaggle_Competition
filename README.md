# How I entered and lost my first Kaggle competition.

[Link to Kaggle competition]([(https://www.kaggle.com/competitions/epf-montpellier-2022/overview/evaluation)](https://www.kaggle.com/competitions/epf-montpellier-2022/overview))

[Link to GitHub repository](https://github.com/GabrielTruong/Teads_Kaggle_Competition)

We entered this competition as a project for the Machine Learning course at our university. Our personnal goal was to learn about the process of building a Machine Learning model and obviously win the competition. 

In this document, we will describe how we build our model and what was our mindset to learn more about Machine Learning.


## 1. Data Collection

The data is available on the [kaggle competition page](https://www.kaggle.com/competitions/epf-montpellier-2022/data) but I decided to try the Kaggle API to get the data directly from the commandline.

``` python
!kaggle competitions download -c epf-montpellier-2022
!unzip epf-montpellier-2022.zip
!rm epf-montpellier-2022.zip
```

Once the data is downloaded, we load the train and test file into a pandas dataframe.
``` python
data_train = pd.read_csv("train.csv")
data_train.head()
```

## 2. Data Preprocessing
In this section, we try to understand the dataset and do the wrangling. We have many variables explained on competition data page.

### **DataSet**

*Data fields* 
- auctionId - a random anonymous id unique to a given auction
- timeStamp - the timestamp (in seconds) of the start of the auction
- placementId - the id of a zone in the web page where the ad would be displayed
- websiteId - the id of the corresponding website
- refererDeepThree - the URL of the page where the ad would be displayed, truncated at its 3rd level and anonymized
- country - the id of the user's country
- operatingSystem - the user's operating system
- browser - the user's browser
- browserVersion - the user's browser version
- device - the user's device type
- environmentType - the type of the webpage environment
- integrationType - the id of the webpage integration type
- articleSafenessCategorization - the categorization of the content of the article

*Target variable*
- isSold - a boolean variable defining wether the auction led to a sale or not

### **Data exploration**

![Check if the data is balanced](./balanced.jpg "Bar plot of isSold")

We can see that the data is balanced. Also the data is either `True` or `False`. This means that later, we will use a classification model. 

### **Data Wrangling**
We prepared the data by removing the columns that we don't need (ie. *auctionId*). Then we took care of the column that needed to such as *timeStamp*. We checked how [others](https://towardsdatascience.com/machine-learning-with-datetime-feature-engineering-predicting-healthcare-appointment-no-shows-5e4ca3a85f96) treated this value. 

After implementing a baseline model (we will talk about modeling later), we had to do feature engineering. So we created new features by concatenating different columns. 

We made sure that these steps was written with function to make it easier to understand and easer to feature engineering with different models. We tried to write docstring for anyone who want to use our model or imitating production type model.
``` python
def concatFeaturesToTarget(df, featuresToConcat, target):
    """ Concatenate features to the target column.
        
        Args:
            df (pandas dataFrame): dataframe to concatenate features to the target column
            featuresToConcat (list): list of features to concatenate
            target (str): name of the target column

        Returns:
            df (pandas dataFrame): dataframe with the features concatenated to the target column
    """
    
    concatDf = pd.DataFrame()
    for feature in featuresToConcat:
        concatDf[feature+"X"+target] = df[target].str.cat(df[feature].astype(str),sep=",")
    return concatDf
``` 

After that, we applied the different functions to our dataset.


## 3. Modeling
A lot of classmate was using Random Forest. Also, we used a lot Random Forest in practical work at school. So we wanted to explore new models. Our Machine Learning teacher advices us to check XGBoost and LightGBM because they were famous for winning Kaggle competitions.

First we split our data into train and test.

``` python
X = data_train_clean.drop(columns=["isSold"]).to_numpy()
y = data_train_clean["isSold"].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
```

### **XGBoost**

NOT FINISHED YET... Update soon on github.

### **LightGBM**

Like XGBoost, we didn't know much about this model. So when we don't know a function or a model, it is always good to search for ressources.
*[Post on Towards Data Science](https://towardsdatascience.com/kagglers-guide-to-lightgbm-hyperparameter-tuning-with-optuna-in-2021-ed048d9838b5
) , [Documentation](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html#tune-parameters-for-the-leaf-wise-best-first-tree)*

Reading these ressources helped us a lot to understand how to use this model and what are the different parameters. Here are the common tips about parameters:

**Hyperparameters that control the tree structure:** 

- In LGBM, the most important parameter to control the tree structure is `num_leaves`. As the name suggests, it controls the number of decision leaves in a single tree. 
- The higher `max_depth`, the more levels the tree has, which makes it more complex and prone to overfit. Too low, and you will underfit. A good range is between 3 and 12.
- Good rule of thumb is: the maximum limit to `num_leaves` should be `2^(max_depth)`
- `min_data_in_leaf`: This is a very important parameter to prevent over-fitting in a leaf-wise tree. Its optimal value depends on the number of training samples and num_leaves. Setting it to a large value can avoid growing too deep a tree, but may cause under-fitting. In practice, setting it to hundreds or thousands is enough for a large dataset.
  

**Hyperparameters for better accuracy:**

- `n_estimators`: controls the number of decision trees 
- `learning_rate`: is the step size parameter of the gradient descent.
- The perfect setup for these 2 parameters (n_estimators and learning_rate) is to use many trees with early stopping and set a low value for `learning_rate`.
- Use large `max_bin` (may be slower)

**Hyperparameters to control overfitting:**
- Use small `max_bin`
- Use small `num_leaves`
- Use `min_data_in_leaf` and `min_sum_hessian_in_leaf`
- Use bagging by `set bagging_fraction` and `bagging_freq`
- Use feature sub-sampling by set `feature_fraction`
- Use bigger training data
- Try `lambda_l1`, `lambda_l2` and `min_gain_to_split` for regularization
- Try `max_depth` to avoid growing deep tree
- Try `extra_trees`
- Try `increasing path_smooth`

We tried different set of hyperparameters to submit the best model. We didn't care much about computing time since LightGBM was the fastest computing model 1.5 times faster than XGBoost. We wanted to have a good accuracy without overfitting too much. This what our model's parameters looked like:

``` python
import lightgbm as lgb 
clf_lgb = lgb.LGBMClassifier(
    max_bin=500,
    max_depth = 8,
    num_leaves = 100,
    min_data_in_leaf=500,
    learning_rate=0.08,
    n_estimators=500
)
```
To find the best parameters, we first tried different combinations of hyperparameters by hand that led us to parameters found above. We will talk later about other methods. 

## 4. Model Evaluation

The evaluation metric for this competition is Mean F1-Score. The F1 score, commonly used in information retrieval, measures accuracy using the statistics precision `p` and recall `r`.

$$Precision = \frac{tp}{tp+fp}$$

$$Recall = \frac{tp}{tp+fn}$$

$$F1 = \frac{2p*r}{p+r}$$

The F1 metric weights recall and precision equally, and a good retrieval algorithm will maximize both precision and recall simultaneously. Thus, moderately good performance on both will be favored over extremely good performance on one and poor performance on the other.

After setting our model parameters, we fit our model. Then we predict the test data and calculate the F1 score. We did that for all the different models.
``` python
clf_lgb.fit(X_train,y_train)
y_pred_lgb = clf_lgb.predict(X_test)
f1_score(y_test,y_pred_lgb)
```
Then, we compare it to the training sample to check if the difference of the two f1 scores is significant. To choose our model, we based our selection on having the best test f1 score and lowest difference between the training and test f1 scores to prevent overfitting.
``` python
y_training_pred_lgb = clf_lgb.predict(X_train)
f1_score(y_train,y_training_pred_lgb)
```

## 5. Submission

We submit our model to the Kaggle competition using the Kaggle API. Our final score on Kaggle was: `0.76377`.
``` python
!kaggle competitions submit -c epf-montpellier-2022 -f testSubmissionFile_bestVersion.csv -m "Trying to get back to top 1 !!"
```

## 6. To go further (but didn't work)

This section describes all the thing we tried but unfortunately didn't work. However, testing other techniques and methods taught us a lot. 

### **Parallel Computing**
First, we wanted to do the competition using `pyspark` but the deadline was too close. So we decided to focus more on the process of implementing a Machine Learning Model. We wanted to try parallel computing so we gave a look to `Dask`.

What `Dask` does is it splits the data into smaller pieces and then it sends these pieces to different workers. The workers then do the work and send the results back to the master. The master then merges the results and sends the final result to the user. It was supposed to accelerate the process of training a model but it didn't work as we wish. So we just use the native version of our models as they offer multi-threading too.

### **Hyperparameter Tuning**
Instead of testing parameters by hand, we also tried `GridSearchCV`. So we read about `GridSearchCV` to find the best parameters. `GridSearchCV` is a class that allows us to find the best parameters for a model by trying every combinaison of the range of parameters you enter. It also performs cross-validation and offer you the best model. However, it takes a lot of time to compute. 

## 7. Conclusion
 We finished second in the competition. We are madly disappointed because our primary goal was to finish top 1. On the other hand, this competition was a great opportunity to learn more about Machine Learning and to try different techniques.

## Sources
- [Ressource about ML process](https://www.kdnuggets.com/2018/05/general-approaches-machine-learning-process.html)
- [Link about XGBoost](https://towardsdatascience.com/xgboost-fine-tune-and-optimize-your-model-23d996fab663)
- [Link about LightGBM](https://towardsdatascience.com/kagglers-guide-to-lightgbm-hyperparameter-tuning-with-optuna-in-2021-ed048d9838b5)
- [Link 2 about LightGBM](https://www.kaggle.com/code/mrtgocer/from-zero-to-hero-lightgbm-classifier/notebook)
- [Link 3 about Light GBM](https://www.kaggle.com/code/prashant111/lightgbm-classifier-in-python/notebook)
- [Link about GridSearchCV](https://www.analyticsvidhya.com/blog/2021/06/tune-hyperparameters-with-gridsearchcv/)
- [Link about Model Selection](https://machinelearningmastery.com/a-gentle-introduction-to-model-selection-for-machine-learning/#:~:text=Model%20selection%20is%20the%20process%20of%20selecting%20one%20final%20machine,SVM%2C%20KNN%2C%20etc.)




