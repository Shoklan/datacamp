# Supervised Learning in R: Classification
## Brett Lantz

# Chapter 1: k-Nearest Neighbors (kNN)
- The sub-domain of **Supervised Learning** focuses on training a machine to learn from prior examples.
- When a concept to be learned is a set of categories then the task is called Classification.
- A **Nearest Neighbor Classifier** takes advantage of the fact that signs look alike.
- It does this by converting the data into a concept of distance and measures the objects distance from one another.
- Many Nearest Neighbor Algorithms use Euclidean Distance:
![Euclidean Distance Formula](images/euclidean-distance.png)
- An algorithm called **K-Nearest Neighbors** uses this idea to classify unlabeled examples.
```r
library(class)
pred <- knn(training_data, testing_data, training_labels)
```
- You can use this as a template for running KNNs:
```r
# Load the 'class' package
library(class)

# Create a vector of labels
sign_types <- signs$sign_type

# Classify the next sign observed
knn(train = signs[-1], test = next_sign, cl = sign_types)
```
- You can aggregate over a dataset using:
```
# Check r10's average red level by sign type
aggregate(r10 ~ sign_type, data = signs, mean)
```
- The letter K in *KNN* specifies the number of neighbors to consider when making the classification.
- The default is `K = 1` in R.
- You may need to play around with this value since it can be important to correct classifications.
- Whichever category shows up the most, is the winner.
- Increasing it doesn't necessarily improve importance.
- **Noise** random information independent of the data can cause fuzzy boundaries.
- In practice, the complexity of the data along with how noisy it is.
- A guideline is to take the square root of the number of observations in the data.
- To actually set the value of K:
```
k_7 <- knn(train = signs[-1], test = signs_test[-1], cl = sign_types, k= 7)
mean(signs_actual == k_7)
```
- Knowing more about the voters' confidence in the classification could allow an autonomous vehicle to use caution in the case there is any chance at all that a stop sign is ahead.
- You can set that by passing `prob = TRUE` inside `knn()`.
- You can then get the votes using `sign_prob <- attr(sign_pred, "prob")`
- While all this is useful, we will need to know how to prepare the data for KNN.
- It is assumed that the data is in numerical format since defining distance between categories is challenging.
- A common solution is to **Dummy Code** indicators to represent these categories; using columns with 1/0 filled in.
- When measuring distance, it is important to keep them within the same scale.
- You'll therefore want to normalize numeric data:
```r
normalize <- function(x){
  return((x - min(x))/ (max(x) - min(x)))
}
```

# Chapter 2: Naive Bayes
- Your phone uses your past locations to attempt to predict your most probable future locations.
- The probability of an event is the number of times something did happen over the number of times it could happen.
- When events have a joint probability, these can be displayed using a Venn Diagram.
![Joint Probability Diagram](images/joint-probability-diagram.png)
- **Independent Events** are when one event does not influence the other event.
- **Dependent Events** are when one event does influence the other event.
- There algorithm **Naive Bayes** applies Bayesian methods to estimate the conditional probability of an outcome.
```R
library(naivebayes)
m <- naive_bayes(lication ~ time_of_day, data = location_history)
```
```R
# Compute P(A)
p_A <- nrow(subset(where9am, location == "office")) / nrow(where9am)

# Compute P(B)
p_B <- nrow(subset(where9am, daytype == "weekday")) / nrow(where9am)

# Compute the observed P(A and B)
p_AB <- nrow(subset(where9am, location == "office" & daytype == "weekday")) / nrow(where9am)

# Compute P(A | B) and print its value
p_A_given_B <- p_AB / p_B
p_A_given_B
```
```R
# Load the naivebayes package
library(naivebayes)

# Build the location prediction model
locmodel <- naive_bayes(location ~ daytype, data = where9am)

# Predict Thursday's 9am location
predict(locmodel, newdata = thursday9am)

# Predict Saturdays's 9am location
predict(locmodel, newdata = saturday9am)
```
```R
# Obtain the predicted probabilities for Thursday at 9am
predict(locmodel, newdata = thursday9am , type = 'prob')

# Obtain the predicted probabilities for Saturday at 9am
predict(locmodel, newdata = saturday9am, type = 'prob')
```
- Adding more predictors complicates the model which is why this model is called **Naive** Bayes.
- This is because it assumes that all events are independent of one another.
- While this is not true in the real world, bending this assumption can still net good results.
- Another problem is that if an event exists but has never been observed then the probability is 0.
- Therefore, anything multiplied by 0 is 0 and this method fails.
- To work around this, is to add a very small number - usually 1 - to each event so nothing can be zero.
- This is called a **Laplace Correction** or **Laplace Estimator**.
- You can change this using:
```R
# Build a new model using the Laplace correction
locmodel2 <- naive_bayes(location ~ daytype + hourtype, data = locations, laplace = 1)
```
- You can also apply this to other problems as well.
- Naive Bayes does not deal well with unstructured or numeric data.
- If you run into these then you will need to to some cleaning first.
- **Binning** is a good way to deal with numeric data.


# Chapter 3: Logistic Regression

# Chapter 4: Classification Trees

# Research:

# Refernce:
