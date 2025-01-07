#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyspark
import pandas as pd
import numpy as np
import math


# In[2]:


from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, StringType, LongType, IntegerType, FloatType
from pyspark.sql.functions import col, column
from pyspark.sql.functions import expr
from pyspark.sql.functions import split
from pyspark.sql import Row
from pyspark.mllib.recommendation import ALS


# In[3]:


ss=SparkSession.builder.appName("Amazon Books ALS-based Recommendation Systems").getOrCreate()


# In[4]:


ss.sparkContext.setCheckpointDir("~/scratch")


# In[12]:


ratings_DF = ss.read.csv("/storage/home/jfl5782/work/project/sample_data/Books_rating_sample0.03.csv"
,header=True, inferSchema=True)


# In[13]:


ratings_DF = ratings_DF.withColumn("rating", col("review/score").cast(FloatType()))


# In[14]:


from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.mllib.recommendation import ALS, Rating
from pyspark.sql.functions import col

ratings_2_DF = ratings_DF.select("User_id", "Title", "rating").na.drop(subset=["User_id", "Title", "rating"])

ratings_2_DF = ratings_2_DF.withColumn("rating", col("rating").cast("double")).na.drop(subset=["rating"])

indexer = StringIndexer(inputCols=["User_id", "Title"], outputCols=["UserId", "BookId"])
model = indexer.fit(ratings_2_DF)
ratings_indexed = model.transform(ratings_2_DF)

ratings_for_als = ratings_indexed.select(col("UserId").cast("int"), col("BookId").cast("int"), col("rating"))

ratings_for_als_rdd = ratings_for_als.rdd

training_RDD, validation_RDD, test_RDD = ratings_for_als_rdd.randomSplit([0.6,0.2,0.2], 19)
training_input_RDD = training_RDD.map(lambda x: (x[0], x[1]) )
validation_input_RDD = validation_RDD.map(lambda x: (x[0], x[1]) )
testing_input_RDD = test_RDD.map(lambda x: (x[0], x[1]) )


# In[15]:


model = ALS.train(training_RDD, 4, seed=17, iterations=30, lambda_=0.1)


# In[16]:


training_prediction_RDD = model.predictAll(training_input_RDD)


# In[18]:


training_target_output_RDD = training_RDD.map(lambda x: ( (x['UserId'], x['BookId']), x['rating'] ) )
training_prediction2_RDD = training_prediction_RDD.map(lambda x: ((x[0],x[1]),x[2]))
training_evaluation_RDD = training_target_output_RDD.join(training_prediction2_RDD)
training_error = math.sqrt(training_evaluation_RDD.map(lambda z: (z[1][0] - z[1][1])**2).mean())
print(f"Training error : {training_error}")



# In[19]:


validation_prediction_RDD = model.predictAll(validation_input_RDD).map(lambda x: ( (x[0], x[1]), x[2] ) )


# In[20]:


validation_evaluation_RDD = validation_RDD.map(lambda y: ( (y[0],y[1]), y[2] )).join(validation_prediction_RDD)
validation_error = math.sqrt(validation_evaluation_RDD.map(lambda z: (z[1][0] - z[1][1])**2).mean())
print(f"Validation Error : {validation_error}")


# In[ ]:


## Initialize a Pandas DataFrame to store evaluation results of all combination of hyper-parameter settings
hyperparams_eval_df = pd.DataFrame( columns = ['k', 'regularization', 'iterations', 'validation RMS', 'testing RMS'] )
# initialize index to the hyperparam_eval_df to 0
index =0
# initialize lowest_error
lowest_validation_error = float('inf')
# Set up the possible hyperparameter values to be evaluated
iterations_list = [15, 30]
regularization_list = [0.1, 0.2, 0.3]
rank_list = [4, 7, 10, 13]
for k in rank_list:
    for regularization in regularization_list:
        for iterations in iterations_list:
            seed = 37
            # Construct a recommendation model using a set of hyper-parameter values and training data
            model = ALS.train(training_RDD, k, seed=seed, iterations=iterations, lambda_=regularization)
            # Evaluate the model using evalution data
            # map the output into ( (userID, movieID), rating ) so that we can join with actual evaluation data
            # using (userID, movieID) as keys.
            validation_prediction_RDD= model.predictAll(validation_input_RDD).map(lambda x: ( (x[0], x[1]), x[2] )   )
            validation_evaluation_RDD = validation_RDD.map(lambda y: ( (y[0], y[1]), y[2] ) ).join(validation_prediction_RDD)
            # Calculate RMS error between the actual rating and predicted rating for (userID, movieID) pairs in validation dataset
            validation_error = math.sqrt(validation_evaluation_RDD.map(lambda z: (z[1][0] - z[1][1])**2).mean())
            # Save the error as a row in a pandas DataFrame
            hyperparams_eval_df.loc[index] = [k, regularization, iterations, validation_error, float('inf')]
            index = index + 1
            # Check whether the current error is the lowest
            if validation_error < lowest_validation_error:
                best_k = k
                best_regularization = regularization
                best_iterations = iterations
                best_index = index - 1
                lowest_validation_error = validation_error
print('The best rank k is ', best_k, ', regularization = ', best_regularization, ', iterations = ',      best_iterations, '. Validation Error =', lowest_validation_error)


# In[18]:


seed = 37
model = ALS.train(training_RDD, best_k, seed=seed, iterations=best_iterations, lambda_=best_regularization)
testing_prediction_RDD=model.predictAll(testing_input_RDD).map(lambda x: ((x[0], x[1]), x[2]))
testing_evaluation_RDD= test_RDD.map(lambda x: ((x[0], x[1]), x[2])).join(testing_prediction_RDD)
testing_error = math.sqrt(testing_evaluation_RDD.map(lambda x: (x[1][0]-x[1][1])**2).mean())
print('The Testing Error for rank k =', best_k, ' regularization = ', best_regularization, ', iterations = ',       best_iterations, ' is : ', testing_error)


# In[19]:


hyperparams_eval_df.loc[best_index]=[best_k, best_regularization, best_iterations, lowest_validation_error, testing_error]


# In[20]:


schema3= StructType([ StructField("k", FloatType(), True),                       StructField("regularization", FloatType(), True ),                       StructField("iterations", FloatType(), True),                       StructField("Validation RMS", FloatType(), True),                       StructField("Testing RMS", FloatType(), True)                     ])


# In[21]:


HyperParams_RMS_DF = ss.createDataFrame(hyperparams_eval_df, schema3)


# In[ ]:


output_path = "/storage/home/jfl5782/work/project/alsResult"
HyperParams_RMS_DF.write.option("header", True).csv(output_path)


# In[ ]:




