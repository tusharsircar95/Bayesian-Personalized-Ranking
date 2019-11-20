# Bayesian-Personalized-Ranking
Bayesian Personalised Ranking or One Class Collaborative Filtering for predicting whether a user reads a particular book. Details explained <a href="https://medium.com/@tusharsircar95/one-class-collaborative-filtering-occf-to-predict-whether-a-user-reads-a-book-286ce31a2d9b">here</a>

Solution mainly comprised of 3 steps:

## Step 1: Trained a latent factor model for one class recommendation (OCCF). 

Defined an embedding U for each userID and embedding B for each bookID.
For each epoch, iterated through Train and for each (userID,bookID) generated N negative samples (userID,bookID’). 
Maximised Sigmoid( Dot(U[userID].B[bookID) - Dot(U[userID].B[bookID’] ) for all pairs.

(Latent Factor Size = 6, Gradient Descent (Adam Optimizer) with LR = 0.01, Lambda = 0.0001, ~250 epochs with LR reduced to 0.001 after 200 epochs, N = 20)

To make predictions for (userID,bookID) in Val, I looked at all other books read by the particular user and calculated the minimum score assigned (say s). I predicted 0 if the score assigned to (userID,bookID) < 0.50*s (if s > 0) or < s (if s < 0). Otherwise, I predicted 1. (Leaderboard score ~ 70-72%)

## Step 2: Extracted features for (userID,bookID) pairs.


Feature Type 1: Look at the Jaccard Similarity between bookID and other books read by this user. Take the mean of the top K highest Jaccard similarities. Do this for K = 1,3,5,7.

Feature Type 2: Similar to Feature Type 1, but calculates Jaccard similarities between users. Do this for K = 1,3,5,7.

Feature Type 3: Popularity of books. Defined as the number of users who read this book divided by the total number of interactions.

Feature Type 4: Activity of users. Defined as the number of books read by this user divided by the total number of interactions.

## Step 3: Boosting - With Logistic Regression

Trained a logistic regression model to predict 0/1 which takes as input the predictions of OCCF and features extracted in step 2. (Leaderboard score ~ 74-75%). This was fit on Val and evaluated on Test to avoid over-fitting (note that all features were scaled to [0,1] for training). 

