Epileptic-Classification

Information about the dataset
Attribute Information:

The original dataset from the reference consists of 5 different folders, each with 100 files, with each file representing a single subject/person. Each file is a recording of brain activity for 23.6 seconds. The corresponding time-series is sampled into 4097 data points. Each data point is the value of the EEG recording at a different point in time. So we have total 500 individuals with each has 4097 data points for 23.5 seconds.

The dataset is divided and shuffled every 4097 data points into 23 chunks, each chunk contains 178 data points for 1 second, and each data point is the value of the EEG recording at a different point in time. So now we have 23 x 500 = 11500 pieces of information(row), each information contains 178 data points for 1 second(column), the last column represents the label y {1,2,3,4,5}.

The response variable is y in column 179, the Explanatory variables X1, X2, ..., X178

y contains the category of the 178-dimensional input vector. Specifically y in {1, 2, 3, 4, 5}:

5 - eyes open, means when they were recording the EEG signal of the brain the patient had their eyes open

4 - eyes closed, means when they were recording the EEG signal the patient had their eyes closed

3 - Yes they identify where the region of the tumor was in the brain and recording the EEG activity from the healthy brain area

2 - They recorder the EEG from the area where the tumor was located

1 - Recording of seizure activity

All subjects falling in classes 2, 3, 4, and 5 are subjects who did not have epileptic seizure. Only subjects in class 1 have epileptic seizure. Our motivation for creating this version of the data was to simplify access to the data via the creation of a .csv version of it. Although there are 5 classes most authors have done binary classification, namely class 1 (Epileptic seizure) against the rest.

Random Forest

Random forest, like its name implies, consists of a large number of individual decision trees that operate as an ensemble. Each individual tree in the random forest spits out a class prediction and the class with the most votes becomes our model’s prediction (see figure below).

Visualization of a Random Forest Model Making a Prediction

The fundamental concept behind random forest is a simple but powerful one — the wisdom of crowds. In data science speak, the reason that the random forest model works so well is:
A large number of relatively uncorrelated models (trees) operating as a committee will outperform any of the individual constituent models.

The low correlation between models is the key. Just like how investments with low correlations (like stocks and bonds) come together to form a portfolio that is greater than the sum of its parts, uncorrelated models can produce ensemble predictions that are more accurate than any of the individual predictions. The reason for this wonderful effect is that the trees protect each other from their individual errors (as long as they don’t constantly all err in the same direction). While some trees may be wrong, many other trees will be right, so as a group the trees are able to move in the correct direction. So the prerequisites for random forest to perform well are:
There needs to be some actual signal in our features so that models built using those features do better than random guessing.
The predictions (and therefore the errors) made by the individual trees need to have low correlations with each other.


Tune Model with Hyper-Parameters

When we used sklearn Decision Tree (DT) Classifier, we accepted all the function defaults. This leaves opportunity to see how various hyper-parameter settings will change the model accuracy. (Click here to learn more about parameters vs hyper-parameters.)

However, in order to tune a model, we need to actually understand it. That's why I took the time in the previous sections to show you how predictions work. Now let's learn a little bit more about our DT algorithm.

Credit: sklearn

Some advantages of decision trees are:

- Simple to understand and to interpret. Trees can be visualized.
- Requires little data preparation. Other techniques often require data normalization, dummy variables need to be created and blank values to be removed. Note however that this module does not support missing values.
- The cost of using the tree (i.e., predicting data) is logarithmic in the number of data points used to train the tree.
- Able to handle both numerical and categorical data. Other techniques are usually specialized in analyzing datasets that have only one type of variable. See algorithms for more information.
- Able to handle multi-output problems.
- Uses a white box model. If a given situation is observable in a model, the explanation for the condition is easily explained by Boolean logic. By contrast, in a black box model (e.g., in an artificial neural network), results may be more difficult to interpret.
- Possible to validate a model using statistical tests. That makes it possible to account for the reliability of the model.
- Performs well even if its assumptions are somewhat violated by the true model from which the data were generated.


The disadvantages of decision trees include:

- Decision-tree learners can create over-complex trees that do not generalize the data well. This is called overfitting. Mechanisms such as pruning (not currently supported), setting the minimum number of samples required at a leaf node or setting the maximum depth of the tree are necessary to avoid this problem.
- Decision trees can be unstable because small variations in the data might result in a completely different tree being generated. This problem is mitigated by using decision trees within an ensemble.
- The problem of learning an optimal decision tree is known to be NP-complete under several aspects of optimality and even for simple concepts. Consequently, practical decision-tree learning algorithms are based on heuristic algorithms such as the greedy algorithm where locally optimal decisions are made at each node. Such algorithms cannot guarantee to return the globally optimal decision tree. This can be mitigated by training multiple trees in an ensemble learner, where the features and samples are randomly sampled with replacement.
- There are concepts that are hard to learn because decision trees do not express them easily, such as XOR, parity or multiplexer problems.
- Decision tree learners create biased trees if some classes dominate. It is therefore recommended to balance the dataset prior to fitting with the decision tree.

Below are available hyper-parameters and defintions:

class sklearn.tree.DecisionTreeClassifier(criterion=’gini’, splitter=’best’, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)

Principle Componenet Analysis (PCA)

- I think it is very cool and good understanding way of PCA.

- Fundemental dimension reduction technique

- It is real life example. I hope it makes more sence for you.
Now lets try one more thing.
- As you remember training time is almost 7 minutes. I think it is too much time. The reason of this time is number of sample(2062) and number of feature(4096).
- There is two way to decrease time spend.

    1. Decrease number of sample that I do no recommend.
    
    2. Decrease number of features. As you see from images all signs are in the middle of the frames. Therefore, around signs are same for all signs.20
    
    3. Think that all of the pixels are feature. Features which are out of the red frame is useless but pixels in red frame take part in training and prediction steps. Therefore, we need to make dimension reduction.
    4. PCA: Principle componenet analysis
        - One of the most popular dimension reduction technique.

        - PCA uses high variances. It means that it likes diversity. For example compare two images above. oUt of red frame there is no diversity (high variance). On the other hand, in red frames there is diversity.

            - first step is decorrelation:
            
             - rotates data samples to be aligned with axes
             
             - shifts data SAmples so they have mean zero
             
             - no information lost
             
             - fit() : learn how to shift samples
             
             - transform(): apply the learned transformation. It can also be applies test data( We do not use here but it is good to know it.)
              
              - Resulting PCA features are not linearly correlated
              
              - Principle components: directions of variance
              
              - Second step: intrinsic dimension: number of feature needed to approximate the data essential idea behind dimension reduction
              
              - PCA identifies intrinsic dimension when samples have any number of features

              - intrinsic dimension = number of PCA feature with significant variance



Lets apply PCA and visualize what PCA says to us.

Feature selection:

Feature selection is also called variable selection or attribute selection.

    -It is the automatic selection of attributes in your data (such as columns in tabular data) that are most relevant to the predictive modeling problem you are working on.

    -feature selection… is the process of selecting a subset of relevant features for use in model construction


    -Feature selection is different from dimensionality reduction. Both methods seek to reduce the number of attributes in the dataset, but a dimensionality reduction method do so by creating new combinations of attributes, where as feature selection methods include and exclude attributes present in the data without changing them.

    -Examples of dimensionality reduction methods include Principal Component Analysis, Singular Value Decomposition and Sammon’s Mapping.

Feature selection using WEKA:
Feature selection in WEKA is divided into two parts:

1. Attribute Evaluator

2. Search Method.
Each section has multiple techniques from which to choose.

The attribute evaluator is the technique by which each attribute in your dataset (also called a column or feature) is evaluated in the context of the output variable (e.g. the class). The search method is the technique by which to try or navigate different combinations of attributes in the dataset in order to arrive on a short list of chosen features.

Some Attribute Evaluator techniques require the use of specific Search Methods. For example, the CorrelationAttributeEval technique can only be used with a Ranker Search Method, that evaluates each attribute and lists the results in a rank order. When selecting different Attribute Evaluators, the interface may ask you to change the Search Method to something compatible with the chosen technique.

Feature selection using Random Forest:

It is pretty common to use model.feature_importances in sklearn random forest to study about the important features. Important features mean the features that are more closely related with dependent variable and contribute more for variation of the dependent variable. We generally feed as much features as we can to a random forest model and let the algorithm give back the list of features that it found to be most useful for prediction. But carefully choosing right features can make our target predictions more accurate .
The idea of calculating feature_importances is simple, but great.
Splitting down the idea into easy steps:
1. train random forest model (assuming with right hyper-parameters)
2. find prediction score of model (call it benchmark score)
3. find prediction scores p more times where p is number of features, each time randomly shuffling the column of i(th) feature
4. compare all p scores with benchmark score. If randomly shuffling some i(th) column is hurting the score, that means that our model is bad without that feature.
5. remove the features that do not hurt the benchmark score and retrain the model with reduced subset of features.
