# [Teads-Kaggle-Competition](https://www.kaggle.com/competitions/epf-montpellier-2022/overview/evaluation)

<h1>Project goal</h1><br>
  
The goal of this prediction model is to predict which auctions would lead to a sale and which auctions would not. By doing so, we can choose to not conduct auctions if we are confident that they would not lead to a sale. This can help us significantly decrease our infrastructure costs.

Before each auction, we dispose of some information about the context. This information can be used as an input to a classification model. The output would be wether this auction would lead to a sale or not.


<h2>Evaluation</h2><br>
The evaluation metric for this competition is Mean F1-Score. The F1 score, commonly used in information retrieval, measures accuracy using the statistics precision p and recall r.

Precision is the ratio of true positives tp to all predicted positives tp + fp. Recall is the ratio of true positives tp to all actual positives tp + fn. 

 
The F1 metric weights recall and precision equally, and a good retrieval algorithm will maximize both precision and recall simultaneously. Thus, moderately good performance on both will be favored over extremely good performance on one and poor performance on the other.

## About Data

- **auctionId** : a random anonymous id unique to a given auction
- **timeStamp** : the timestamp (in seconds) of the start of the auction
 
- **placementId** : the id of a zone in the web page where the ad would be displayed
- websiteId : the id of the corresponding website
- **refererDeepThree** : the URL of the page where the ad would be displayed, truncated at its 3rd level and anonymized
- **country** : the id of the user's country
- **operatingSystem** : the user's operating system
- **browser** : the user's browser
- **browserVersion** : the user's browser version
- **device** : the user's device type
- **environmentType** : the type of the webpage environment
- **integrationType** : the id of the webpage integration type
- **articleSafenessCategorization** : the categorization of the content of the article

### TODO
**Goal**
- Learn PySpark
- Write an article "How we entered our first Kaggle Competition and how we won it"
- Think about how we'll write this article

**ML Process**

[Ressource about ML process](https://www.kdnuggets.com/2018/05/general-approaches-machine-learning-process.html)
- Data Collection: Set-up Kaggle API to get datasets directly from commandline
- Data Preparation: Clean the data (Sanitize/Format/Normalization/TrainTestSet)
- Choose a Model: Classification so state of the art of classification model

- Set up deadlines

Notes for article
- Décision (pas de spark)
- One Hot encoding trop coûteux ? -> TargetEncoding
- Traitement du timestamp
- Refactor en fonction
- Classification: Random Forest, XGBoost

.unique => récup tous les solds et les réattribuer après 
Pas le même nombre de colonnes dans le test avec get_dummies => utiliser uniquement target encoding ? 
Valeur manquante du target encoding => remplacer par np.nan puis KNN Impute ?

- GridSearchCV sur 3 models: LightGM, XGBoost, Random Forest
- CrossValidation avec KFold
- Replace Target Encoding puis KNN Impute finalement pas à faire car on utilise .transform
- Ecrire un article sur le processus entier (Data Collection, Data Wrangling, Model Selection...)

https://towardsdatascience.com/xgboost-fine-tune-and-optimize-your-model-23d996fab663
https://www.analyticsvidhya.com/blog/2021/06/tune-hyperparameters-with-gridsearchcv/
https://machinelearningmastery.com/a-gentle-introduction-to-model-selection-for-machine-learning/#:~:text=Model%20selection%20is%20the%20process%20of%20selecting%20one%20final%20machine,SVM%2C%20KNN%2C%20etc.)

https://towardsdatascience.com/kagglers-guide-to-lightgbm-hyperparameter-tuning-with-optuna-in-2021-ed048d9838b5
https://www.kaggle.com/code/mrtgocer/from-zero-to-hero-lightgbm-classifier/notebook
https://www.kaggle.com/code/prashant111/lightgbm-classifier-in-python/notebook