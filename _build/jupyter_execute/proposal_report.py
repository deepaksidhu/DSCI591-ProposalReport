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
# The Web Analytics Team of the Government of Canada (GoC) is seeking to supplement their current Adobe Analytics (AA) workflows with data science techniques. To tackle their questions, our team will provide an overview of available data science techniques related to web services, and mainly focus on three techniques: time series forecasting, prediction on service satisfaction, and survey clustering.  The final report will highlight the value added by supplementing AA insights with data science techniques. The final deliverable is a Jupyter Book providing documentation for the application of the selected data science techniques.
# 

# ### 2. Introduction
# 
# 
# 

# The Web Analytics Team currently leverages AA to understand and analyze web traffic data ({numref}`adobe-image`). Although AA is a powerful tool, it only covers a small portion of data science techniques. Here are some examples of data science techniques that are not part of the scope of AA: time series forecasting, prediction of service satisfaction, survey clustering, visitor segmentation, recommender systems, Chatbot, and A/B testing.
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
# Sample data will be exported from AA to demonstrate the workflow and benefits of data science techniques. Depending on the tasks, different dimensions, metrics and custom segements will be extracted from the data warehouse. The objectives of this project are 1) overview of available data science techniques applicable to web services; 2) demonstration of three useful data science techniques: time series forecasting, prediction on service satisfaction, and survey clustering; 3) showing the benefits and value-added by employing data science techniques to the data collected by AA. The deliverable of this project will be a Jupyter book including documentation, three models with processed datasets, and well-documented scripts written in Python.
# 

# ### 3. Data Science Techniques
# 

# #### 3.1 Jupyter Book
# 
# Jupyter Book is an open-source tool for producing quality books based on computational material and data science techniques implemented using the Python language {cite}`van1995python`. The main advantage of Jupyter Book over AA is that it provides a clear and reproducible workflow for the analysis of data. We will also explore the options of getting direct data access from AA through Python wrapper {cite}`aanalytics2`. The following sections will describe the techniques in more detail.
# 

# #### 3.2 Time Series Forecasting
# 
# 2 Million average unique visitors come to the GoC's website daily to apply for visas, to browse taxes, and navigate benefits pages. These are the top few reasons to come on the website.  {numref}`visitor-traffic` below shows the  traffic trend for the last 2 years:
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
# According to www.similarweb.com, GoC is one of the top 20 websites visited in Canada. Hence, it’s useful to predict how many visitors would be visiting GoC's website daily. The data science technique of **time series forecasting** will be used to predict traffic on the GoC's website. Furthermore, there seems to be a positive trend in the daily website traffic. It started from 1M and reached 4M by the end of April 2021. Also, there is a strong seasonal signal with period of 7 days. The traffic is down on weekend and is up every Monday's. Also, there is a strong influence of external events such as there was a peak on 10/21/2019 because of the Canafian elections.
# 
# Baseline models such as **ARIMA** is considered first and then moving to machine learning model such as **recursive forecasting** to predict traffic. {numref}`time-series-model` below shows the workflow for training a time series model on web traffic data {cite}`timeseries`.
# 
# ```{figure} img/funnel.PNG
# ---
# height: 300px
# name: time-series-model
# ---
# Time series workflow for the daily web traffic of visitors to the Government of Canada website
# ```
# 
# During the US election 2016, GoC website became temporarily inaccessible to users as a result of significant increase in the traffic volume. *It's bad for business, for the company's image, and users' experience*. Hence, model predicting web traffic will not only help in improving the server's performance but also in managing internal resources for website maintenance. 
# 
# ```{figure} img/page-load-error.PNG
# ---
# height: 300px
# name: load-error
# ---
# 
# ```
# 
# Due to the seasonality and irregular data, prediction intervals would be used to quantify the forecast uncertainty.
# 
# 
# 

# #### 3.3 Prediction on Web Satisfaction 
# 
# A user satisfaction survey is offered to 5% of the daily visitors. Because this is a small proportion it is worthwhile to predict the satisfaction level of the other visitors based on their behaviors on the GoC's website. These behaviors and attributes include geographic information, time spent on pages, referral page sources, and etc. The visiting behaviors will be combined as features, and an empirical model will be proposed and evaluated.  As a result, the model  ({numref}`prediction-satisfaction`) will predict what the probable satisfaction is, and the feature importances in the variations.
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
# Studies on such quality of experience (QoE) have been done massively in profit-driven contexts {cite}`suchanek2018customer,ballestar2019predicting`. This project on GoC's website will enhance and enrich the operation and management in GoC by using machine learning techniques to **predict satisfaction on web services**. Simple models would be implemented first, such as logistic regression, random forest in the binary prediction phase, followed by a multilayer perceptron (MLP) artificial neural network (ANN). Furthermore, page referring sequences will be parsed as sequential data to be applied in an LSTM model. To evaluate the model’s performance, the assessment criteria include accuracy, precision, recall, and F-1 score. In phase two as multiclass prediction, RMSE is proposed to use as the starting metric.
# 
# 
# The findings of this predicting model involve two parts: the first part is the probable prediction of website satisfaction given a visit; the second is what behaviors are related to lower or higher satisfaction results. This will be helpful for the web services team, to better understand visitors’ behavior and insights towards web service redesign and maintenance. However, possible limitations are the interpretation of the web behaviors, model implementation, imbalanced training set, incorporation and processing of open text data.
# 

# #### 3.4 Survey Clustering
# 
# The GoC's task survey has text feedback field of visitor’s experience navigating through the web pages. {numref}`feedback-examples` displays some examples of the visitor feedback:
# 
# ```{figure} img/survey.png 
# ---
# height: 250px
# name: feedback-examples
# ---
# Example of text feedback of visitors based on the experience of naviagating through the web pages
# ```
# 
# The data science technique of **Topic Modelling** {cite}`blei2003modeling` will be used to cluster the topic of the text response of the visitor’s browsing experience. Topic Modelling with **Latent Dirichlet Allocation (LDA)** {cite}`944937` will be employed to train the text data and to extract clusters of topics that are dominant in the text feedback. Each topic cluster determined by the LDA model would be labeled by the major sentiment expressed in the word distribution that makes up the topic. An illustration of assigning a topic to visitor feedback is demonstrated below:
# 
# The trained LDA model assigns the feedback into the **Technical Issues** bucket in {numref}`technical-bucket`.
# 
# ```{figure} img/technical-bucket.png 
# ---
# height: 150px
# name: technical-bucket
# ---
# An example of techinical concern expressed by the visitor and LDA model classification
# ```
# 
# For the next visitor feedback, the LDA model assigns the feedback to the **Complex Flow** bucket in {numref}`complex-bucket`.
# 
# ```{figure} img/Complex-bucket.png 
# ---
# height: 130px
# name: complex-bucket
# ---
# An example of complex flow issue expressed by the visitor and LDA model classification
# ```
# 
# The clustering technique with LDA would provide an automated workflow of labeling the topic expressed in the text feedback. Hence, the web services team can use these topics as insights for application and optimize the delivery of web services.
#  
# However, there are several limitations with the text data. First, there is only 3800 text feedback available for analysis. Hence, training a sophisticated clustering model would be a challenge. Another limitation is the complexity of cleaning and preprocessing text. Lastly, LDA is an unsupervised machine learning approach, hence we don’t have the correct notion of how many major topics are present in the text feedback. The team will deal with this issue by comparing the similarity scores of the topic-word distribution for different LDA models. In addition, we don’t have ground truth labels for the main topic of the visitors' feedback.
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
