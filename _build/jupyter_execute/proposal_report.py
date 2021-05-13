#!/usr/bin/env python
# coding: utf-8

# # Data Science Meets Government of Canada Web Services
# 
# ***By Anita Li, Deepak Sidhu, Jianru Deng, Sakshi Jain***<br>
# *May 14th, 2021*
# 
# 

# ## Proposal Report
# 
# ### 1. Executive Summary
# 
# 
# 

# 
# The Web Analytics Team of the Government of Canada (GoC)  is seeking advice on available data science techniques that can be added to supplement their current Adobe Analytics (AA)  workflow. To tackle their questions, our team will provide an overview of available data science techniques related to web services, and mainly focus on three techniques that provide the most value: time series forecasting, prediction on service satisfaction, and survey clustering. By the end of the project, we hope to show the benefits and value-added by employing data science techniques other than the superficial insights offered by AA. The project will run for eight weeks and the final deliverable is a Jupyter book including documentation and application of data science techniques.

# ### 2. Introduction
# 
# 
# 

# The Web Analytics Team  is responsible for understanding and analyzing web traffic data, and currently they are leveraging AA to do this job. AA is a powerful tool to collect, process, and report web service data. It has many visualization options, making it easier to visualize and report data ({numref}`adobe-image`). However, AA has its limitations as it only covers a small portion of data science techniques. Here are some examples of data science techniques that are not part of the scope of AA: time series forecasting, prediction of service satisfaction, survey clustering, visitor segmentation, recommender systems, Chatbot, and A/B testing.
# 
# ```{figure} img/adobe-eda.png
# ---
# width: 600px
# height: 300px
# name: adobe-image
# ---
# Adobe Analytics dashboard showing various visualizations options for displaying web traffic
# ```
# 
# 
# The purpose of this project is to provide a demonstration to the Web Analytics team on what and how data science techniques could be used to their web service data, as well as the added value to their current workflow. The dataset used in this project will be exported from AA. Depending on the specific tasks, we may extract different date ranges and metrics from the data warehouse. The objectives of this project are 1) overview of available data science techniques related to web services; 2) demonstration on three typical data science techniques: time series forecasting, prediction on service satisfaction, and survey clustering; 3) showing the benefits and value-added by employing data science techniques in addition to those offered by AA. The deliverables of this project include documentation using Jupyter book, three models with the processed dataset, and well-documented scripts written in Python.

# ### 3. Data Science Techniques
# 

# #### 3.1 Jupyter Book
# 
# Jupyter Book is an open-source tool for producing quality books based on computational material and data science techniques implemented using the Python language {cite}`van1995python`. The main advantage of Jupyter Book over AA is that it provides a clear and reproducible workflow for the analysis of data. We will also explore the options of getting direct data access from AA through Python wrapper {cite}`aanalytics2`. 
# 
# The Jupyter Book provided to the web services team will contain the documentation and workflow of the common data science techniques applicable to the data. The web services team of GoC will be able to adapt the techniques discussed in the Jupyter Book to their workflow and gain useful insights to improve and optimize their delivery of web services. The following sections will talk about the components of the Jupyter Book.
# 

# #### 3.2 Time Series Forecasting
# 
# 2 Million average unique visitors come to the GoC's website daily to apply for visas, to browse taxes and benefits pages  or to apply for jobs. These are the top few reasons to come on the website.  {numref}`visitor-traffic` below shows the unique visitors (traffic) trend for last 2 years and 4 months on the site:
# 
# 
# ```{figure} img/visitors-trend.PNG
# ---
# height: 300px
# name: visitor-traffic
# ---
# Daily Web Traffic of unique visotrs coming to the Government of Canada website
# ```
# 
# According to www.similarweb.com, GoC is one of the top 20 websites visited in Canada. Hence, It’s useful to predict how many visitors would be visiting GoC's website daily.The data science application of **time series forecasting** will be used to predict traffic on the GoC's website. There are 3 main patterns in any time series: Trend (long term increases or decreases in the series) , Seasonality (a regular variation) and Cyclicity (variations in the series that repeat with some regularity but of unknown and changing period).
# 
# There seems to be a positive trend in the website traffic. It started from 1M and reached to 4M by the end of April 2021. Also, there is a strong seasonal signal with a 7 days period. The traffic is going down every weekend and then reaching the top every Monday. There is no cyclicity. However, there is a strong influence of external events such as there was a peak on 10/21/2019 because of Canadian election day.
# 
# To implement time series, it is important to remove any trend, seasonality and cyclicity to understand data better. Baseline models such as **ARIMA** models are a good start and then move to complicated models such as **recursive forecasting** in Machine learning to predict traffic. To evaluate the model, the mean absolute scaled error (MASE), Symmetric mean absolute percentage error (SMAPE) etc. will be used. {numref}`time-series-model` below shows the workflow for training a time series model on web traffic data {cite}`timeseries`.
# 
# 
# 
# ```{figure} img/funnel.PNG
# ---
# height: 300px
# name: time-series-model
# ---
# Time series workflow for the daily web traffic of visitors to the Government of Canada website
# ```
# 
# 
# 
# During US election 2016, the Immigration, Refugees and Citizenship website became temporarily inaccessible to users as a result of a significant increase in the volume of traffic. It's bad for business. It's bad for the company's image and it's a bad users' experience. Hence, predicting web traffic will help not only in improving the server or the web site performance but also in managing internal resources such as website maintenance. 
# 
# ```{figure} img/page-load-error.PNG
# ---
# height: 300px
# name: load-error
# ---
# 
# ```
# 
# 
#  
# Forecasting is very challenging as we’re trying to predict the future. Due to the seasonality and  irregular data, including external regressors, prediction intervals would be used to quantify the uncertainty.
# 
# 

# #### 3.3 Prediction on Web Satisfaction 
# 
# Given the fact that there is a very limited number (5%) of visitors who are invited to the survey, it is worthwhile to predict their satisfaction level based on their online behaviors on the GoC website. Those behaviors or the attributes could include geographic information, time spent on particular pages, and referral page sources etc. The prediction model will combine those visiting behaviors as features, and an empirical model will be evaluated and proposed.  As a result, the model ({numref}`prediction-satisfaction`) will predict what the probable satisfaction is, and what role a feature plays in those variations.
# 
# 
# ```{figure} img/prediction-satisfaction.png 
# ---
# height: 200px
# name: prediction-satisfaction
# ---
# An illustration of a prediction model with input of various online behaviors and outputs the probable satisfaction levels.
# ```
# 
# 
# Studies on such quality of experience (QoE) have been done massively in a profit-driven context {cite}`suchanek2018customer,ballestar2019predicting`. This project on GoC website will enhance and enrich the operation and management in GoC by using machine learning techniques to **predict satisfaction on web services**. Simple models are proposed initially as Logistic regression, Random Forest and lgbm in binary prediction phase, followed by a multilayer perceptron (MLP) artificial neural network (ANN). Furthermore, page referring sequences will be parsed as sequential data to be applied in an LSTM model. To evaluate the model’s performance, the assessment criteria includes accuracy, precision, recall and F-1 score, when it is set to be a binary prediction in phase one. In phase two as multiclass prediction, RMSE is proposed to use as the first metrics.
# 
# The findings of this particular data science application involve two parts: the first part is the probable prediction of website satisfaction of one visit; the second is what behaviors are related to lower or higher satisfaction results. This will be helpful for the web analytics team, and moreover public sectors as such to better understand visitors’ behavior and the possible direction towards web service redesign and maintenance. However, possible limitations are the interpretation of the web behaviors and the model implementation, the imbalanced training set, the possible incorporation and processing of open text data.
# 
# 

# #### 3.4 Survey Clustering
# 
# One of the fields of the GoC Task Survey Success is text feedback based on the visitor’s experience navigating through the web pages. {numref}`feedback-examples` displays some examples of the visitor feedback:
# 
# ```{figure} img/survey.png 
# ---
# height: 250px
# name: feedback-examples
# ---
# Example of text feedback of visitors based on the experience of naviagating through the web pages
# ```
# 
# The data science technique of **Topic Modelling** {cite}`blei2003modeling` will be used to cluster the text response of the visitor’s browsing experience. The importance of clustering the text feedback is to determine the topic inherent in the visitor’s response. Topic Modelling with **Latent Dirichlet Allocation (LDA)** {cite}`944937` will be used to train the text data, and to extract clusters of topics that are dominant in the text feedback. Each topic cluster determined by the LDA model would be labeled by the major sentiment expressed in the distribution of the words that makes up the topic. An illustration of assigning a topic to visitor feedback is demonstrated below:
# 
# The visitor's response is concerning the broken links, and HTTP errors encountered navigating through the web pages. Hence the trained LDA model assigns it to **Technical Issues** bucket in {numref}`technical-bucket`.
# 
# ```{figure} img/technical-bucket.png 
# ---
# height: 150px
# name: technical-bucket
# ---
# An example of techinical concern expressed by the visitor and LDA model classification
# ```
# 
# For the next example we see that the visitor had difficulty following the web page. Hence the LDA model assigns the feedback to the **Complex Flow** bucket in {numref}`complex-bucket`.
# 
# ```{figure} img/Complex-bucket.png 
# ---
# height: 130px
# name: complex-bucket
# ---
# An example of complex flow issue expressed by the visitor and LDA model classification
# ```
# 
# The clustering technique with LDA would provide an automated workflow of labeling the topic expressed in the feedback of visitors. This is of significant importance since the most pressing issues of the visitor's experiences will be highlighted to the Web Services Team and they can use this as insights for application, to improve and provide better web services.
# 
# The text feedback field is useful for training a clustering model with LDA, however, there are several limitations with this data. Since the GoC started collecting this data in January of this year, we only have 3800 visitor text feedback. Hence training a sophisticated clustering model would be a challenge. Another limitation is the complexity of cleaning and preprocessing text data for modeling. Lastly, LDA is an unsupervised machine learning approach, hence we don’t have the correct notion of how many major topics are present in the text feedback. The team will deal with this issue by comparing the similarity scores of the topic-word distribution for different LDA models. In addition, we don’t have ground truth labels for the main topic of the visitors' feedback.
# 
# 

# 
# ### 4. Timeline
# 

# | deliverables               | week1 | week2 | week3 | week4 | week5 | week6 | week7 | week8 |
# |----------------------------|-------|-------|-------|-------|-------|-------|-------|-------|
# | proposal presentation      | x  |       |       |       |       |       |       |       |
# | proposal report            | x | x |       |       |       |       |       |       |
# | jupyter book               |       |   x    |     x  |    x   |   x    |  x     |  x     |       |
# | time series forecasting    |       |    x   |     x  |    x   |    x   |  x     |       |       |
# | prediction on satisfaction |       |    x   |     x  | x      |     x  |  x     |       |       |
# | survey clustering          |       |     x  |    x   | x      |     x  |  x     |       |       |
# | final presentation         |       |       |   x    |  x     |     x  |   x    |   x    |       |
# | final report               |       |       |    x   |  x     |     x  |   x    |   x    |    x   |
# <div align="center"> <strong>Fig 8:</strong> Timeline for the project task and deliverable</div>

# Milestone tasks from week 3 to week 6.
# 
# | Milestones              | Due Date|Tasks | 
# |----------------------------| ------   | ------- |
# | Milestone 1    | Week 3| preliminary modeling with simple model trails, and simple metrics  |
# | Milestone 2    | Week 4| research and add extra possible modeling and optimization, proper metrics  |
# | Milestone 3    | Week 5| feature selection and possible feature engineering  |
# | Milestone 4    | Week 6| interpretation and more optimization, thinking of limitations  |
# <div align="center"> <strong>Fig 9:</strong> Milestone tasks from Week 3 to Week 6</div>

# ## 5. Bibliography
# 
# ```{bibliography} references.bib
# ```
