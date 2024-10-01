import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

# def create_preprocessor(categorical_features, numerical_features):
#     # Define the column transformer
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', StandardScaler(), numerical_features),
#             ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
#         ]
#     )
#     return preprocessor  # Return the preprocessor


def create_numerical_preprocessor(numerical_features):
    preprocessor = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # or 'median', 'most_frequent'
        ('scaler', StandardScaler())
    ])
    return preprocessor

# Create a preprocessor for categorical features (if needed)
def create_categorical_preprocessor(categorical_features):
    preprocessor = OneHotEncoder(handle_unknown='ignore')
    return preprocessor

# Dictionary of regression models (Add this to Main script)
# models = {
#     'Linear Regression': LinearRegression(),
#     'Lasso Regression': Lasso(alpha=0.1),
#     'Ridge Regression': Ridge(alpha=1.0),
#     'Random Forest Regression': RandomForestRegressor(),
#     'Support Vector Regression': SVR(),
#     'Elastic Net': ElasticNet(),
#     'Decision Tree Regression': DecisionTreeRegressor(),
#     'Gradient Boost Regression': GradientBoostingRegressor(),
#     'KNeighbors Regression': KNeighborsRegressor()
# }

def create_pipelines(preprocessor, models):
    # Create pipelines for each model
    pipelines = {name: Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ]) for name, model in models.items()}
    return pipelines  # Return the created pipelines

# ----------------------------------------------------------------------------------
# Performance Metrics
# ----------------------------------------------------------------------------------

def evaluate_models(pipelines, X_train, y_train, X_test, y_test):
     # Print shapes and types for debugging
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)
    
    # Dictionary to hold performance metrics
    performance_metrics = {}

    # Evaluate each model
    for name, pipeline in pipelines.items():
        # Fit the model
        pipeline.fit(X_train, y_train)

        # Calculate training and testing scores
        train_score = pipeline.score(X_train, y_train)
        test_score = pipeline.score(X_test, y_test)
        
        # Make predictions
        predictions = pipeline.predict(X_test)
        
        # Calculate performance metrics
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        # Store the metrics
        performance_metrics[name] = {
            'Training Score': train_score,
            'Testing Score': test_score,
            'Mean Absolute Error': mae,
            'Mean Squared Error': mse,
            'R² Score': r2
        }

    # Convert the performance metrics to a DataFrame
    performance_df = pd.DataFrame(performance_metrics).T

    # Sort the DataFrame by Mean Absolute Error (MAE) from highest to lowest
    performance_df_sorted = performance_df.sort_values(by='Mean Absolute Error', ascending=False)

    # Display the performance metrics in a single line
    print(performance_df_sorted.to_string(index=True))
    
    return performance_df_sorted  # Optionally return the sorted DataFrame for further use
    
# ----------------------------------------------------------------------------------
# Hyper Parameter Tuning Using Grid Search
# ----------------------------------------------------------------------------------

def tune_models(pipelines, param_grids, X_train, y_train):
    results = {}
    for name, pipeline in pipelines.items():
        print(f"Tuning {name}...")
        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grids[name], cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        results[name] = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }
    return results

# ----------------------------------------------------------------------------------
# Fit the Best Parameters in the Models
# ----------------------------------------------------------------------------------

def create_and_fit_model(model_name, best_params, X_train, y_train, pipelines):
    # Retrieve the pipeline for the specified model
    pipeline = pipelines[model_name]
    
    # Update the regressor in the pipeline with the best parameters
    for param, value in best_params.items():
        setattr(pipeline.named_steps['regressor'], param.split('__')[1], value)
    
    # Fit the model on the training data
    pipeline.fit(X_train, y_train)
    
    return pipeline  # Return the fitted pipeline