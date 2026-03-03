# **FAKE NEWS DETECTOR**
## Context

The rapid growth of online media has led to an increase in the spread of fake and misleading news. Detecting fake news manually is challenging due to the large volume of content published daily. This project focuses on building a Machine Learning model to automatically classify news articles as REAL or FAKE based on their textual content.

The dataset consists of labeled political news articles. The objective is to train a text classification model and integrate it into a web application for real-time prediction.

## Content

The dataset contains two categories of news articles:

1. REAL – Genuine political news articles.

2. FAKE – Fabricated or misleading political news articles.

Each record contains the main text of the news article used for classification.

The project workflow includes:

1. Loading and merging the REAL and FAKE datasets.

2. Assigning labels to each article.

3. Splitting the dataset into training and testing sets (80% / 20%).

4. Converting text data into numerical format using TF-IDF vectorization.

5. Training a Logistic Regression classifier.

6. Evaluating the model using accuracy score (98.63%).

7. Saving the trained model and integrating it into a Flask web application for real-time prediction.
