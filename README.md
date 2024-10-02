<div id="header" align="center">

  # Garment Worker Productivity Prediction  
</h1>
</div>

<div id="header" align="center">

</div>

## _Project Overview_

This project aims to predict the productivity of garment employees based on features such as working hours, incentives, overtime, and more. The goal is to develop machine learning models that can accurately predict the continuous productivity score for garment workers, helping factories understand key drivers of productivity and optimize their operations. 

## _Dataset_ 

The dataset was retrived from https://archive.ics.uci.edu/dataset/597/productivity+prediction+of+garment+employees
It was called Productivity Prediction of Garment Employees and donated on 8/2/2020.

The dataset used for this project is Garment Worker Productivity, which consists of 1,197 rows and 15 features, including:

  •	date
  
  •	quarter
 
•	department
 
•	day

•	team

•	targeted_productivity

•	smv (Standard Minute Value)

•	wip (Work In Progress)

•	over_time

•	incentive

•	idle_time

•	idle_men

•	no_of_style_change

•	no_of_workers

•	actual_productivity


 
The target variable, actual_productivity, is continuous and is the primary variable we aim to predict using various machine learning models. 

## _Steps of a Regression Model for the Dataset:_

1. Read in the DataFrame: Load the dataset into a pandas DataFrame.
2. Determine X and y: Identify the feature variables (X) and the target variable (y).
3. Split the Data: Use train_test_split() to divide the data into training and testing sets.
4. Scale the Numerical Columns: Apply a scaling method (like StandardScaler) to the numerical columns of the training set, and then use the same scaler to transform the testing set.
5. Fit the Regression Model: Create and fit the various regression models (e.g., linear regression, SVR, random forest) using the scaled training data.
6. Check Performance Metrics: Evaluate the model's performance using appropriate metrics, such as Mean Absolute Error (MAE), Mean Squared Error (MSE) and R² score.

## _Data Cleaning and Preprocessing?_

1. Handling Missing Values

	•	The dataset had missing values in some columns (e.g., wip). We handled these by filling missing numeric values with the mean of the respective column.

	•	For missing target variable (actual_productivity), rows with missing target values were dropped. 

2. Feature Selection

	•	We removed targeted_productivity and actual_productivity from the features to avoid data leakage.

	•	The column date was also removed as it did not provide relevant information for prediction.

3. Categorical Feature Encoding

	•	One-Hot Encoding was applied to categorical variables such as quarter, department, and day using OneHotEncoder from scikit-learn. This transformed these categorical variables into numerical representations suitable for machine learning.

4. Scaling Features

	•	Missing values in the features were filled using the mean, and all numeric features were scaled for better model performance.

## _Model Training_

The first run through of the dataset after cleaning and filling the NaN values with mean values is shown below.
We trained several machine learning models to predict the continuous actual_productivity score:

1. Support Vector Regressor (SVR)

•	A linear Support Vector Regressor was used to model the relationship between features and productivity.

•	Model Performance:

      •	 R² Score: 0.32
     
      •	 Mean Absolute Error (MAE): 0.096


2. Decision Tree Regressor

•	A Decision Tree Regressor was trained to predict the continuous productivity score.

•	Model Performance:

      •	 R² Score: 0.31

      • Mean Absolute Error (MAE): 0.08

3. Random Forest Regressor

•	A Random Forest Regressor with 200 estimators was used for better predictive power.

•	Model Performance:

      •       R² Score: 0.48
     
      •       Mean Absolute Error (MAE): 0.075

4. K-Nearest Neighbors Regressor (KNN)

•	The KNN regressor was tested with different values of k to optimize performance.

•	Model Performance:

	   •	  R² Score: 0.39
 
	   •	  Mean Absolute Error (MAE): 0.086
    
5. Gradient Boosting Regressor

•	A Gradient Boosting Regressor was implemented to improve prediction accuracy by combining several weak learners (decision trees) into a strong model. We used 100 estimators and a learning rate of 0.2.

•	Model Performance:

	  •	  R² Score: 0.43
 
	  •	  Mean Absolute Error (MAE): 0.082

6. Lasso Regressor

•	Model Performance:

	  •	  R² Score: - 0.000685
 
	  •	  Mean Absolute Error (MAE): 0.13

7. Elastic Net Regressor

•	Model Performance:

	  •	  R² Score: - 0.000685
 
	  •	  Mean Absolute Error (MAE): 0.13

8. Ridge Regressor

•	Model Performance:

	  •	  R² Score: 0.188
 
	  •	  Mean Absolute Error (MAE): 0.11

9. Ridge Regressor

•	Model Performance:

	  •	  R² Score: 0.188
 
	  •	  Mean Absolute Error (MAE): 0.11

## _Results_

At the beginning the Regression Models applied to the dataset with the WIP column modification of the NaN values to be replaced with the mean value of the column, did not have the best scores for any of the models. All the testing scores were below 50%. Even after applying the Hyper Parameter Tuning by use of the Grid Search function, the scores did no better. The anaylsis was revisted and tested with the NaN values being zero which still did not improve the testing metric scores. Lastly we tried dropping the NaN values along with rows associated with them. This did drop our number of instances down by almost half. Then with rerunning the models again with NaN values dropped, the testing metric scores came up to more reasonable values. The Gradient Boosting Regressor and Random Forest Regressor provided the best performance, with the highest R² score and the lowest error rate after cleaning the data by dropping the NaN instances.

Gradient Boost Regressor

•	A Gradient Boosting Regressor was implemented to improve prediction accuracy by combining several weak learners (decision trees) into a strong model.

•	Model Performance:

	  •	  R² Score: 0.851
 
	  •	  Mean Absolute Error (MAE): 0.0348

Random Forest Regressor

•	Model Performance:

      •       R² Score: 0.831
     
      •       Mean Absolute Error (MAE): 0.0337

## _Conclusion_

This project demonstrates the use of machine learning to predict garment worker productivity. The Gradient Boosting Regressor and Random Forest Regressor achieved the best overall performance, providing valuable insights into which factors most strongly influence productivity in garment factories. We originally had issues with the data and fitting any models to the dataset. All the models produced poor results of less than 50% which was worse than flipping a coin. That means machine learning had no useful purpose for the data set that we were working on. After revisiting the dataset and dropping the NaN values that was present in the dataset, the testing scores and error greatly improved as mentioned in the Results section above. Two models produced testing scores of about 83%-85% while only having an error rate of about 3%.

## _References_

•	Dataset: Garment Worker Productivity Data. https://archive.ics.uci.edu/dataset/597/productivity+prediction+of+garment+employees

•	Google Searches: Used for researching best practices for handling missing values and tuning model hyperparameters. Used http://fliki.ai/ .

•	XpertLearning: Provided foundational knowledge on regression techniques and machine learning model evaluation.

•	TA and Instructor Guidance: Assistance from the teaching assistant and instructor was very helpful in the understanding of feature engineering, model selection, and improving model performance.



## _Credits_ :thumbsup:

Tuan Huynh, Jaiden Flint, Kavita Gopal
