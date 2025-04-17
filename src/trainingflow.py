
# src/trainingflow.py
from metaflow import FlowSpec, step, Parameter, current
import pandas as pd
import numpy as np
import os

class WineQualityTrainingFlow(FlowSpec):
    """
    A flow for training models on the Wine Quality dataset and registering the best model
    with MLflow.
    """
    # Define parameters for the flow
    data_path = Parameter('data_path', 
                          default='~/usf/mlops-course/data/wine-quality.csv',
                          help='Path to the wine quality dataset')
    test_split = Parameter('test_split', 
                          default=0.2, 
                          help='Proportion of data to use for testing',
                          type=float)
    random_state = Parameter('random_state', 
                             default=42, 
                             help='Random seed for reproducibility',
                             type=int)
    cv_folds = Parameter('cv_folds', 
                         default=5, 
                         help='Number of cross-validation folds',
                         type=int)

    @step
    def start(self):
        """
        Start the flow by loading and preprocessing the data
        """
        import pandas as pd
        from sklearn.model_selection import train_test_split
        
        # Load data
        print(f"Loading data from {self.data_path}...")
        try:
            data_path = os.path.expanduser(self.data_path)
            data = pd.read_csv(data_path, sep=';')
        except FileNotFoundError:
            print(f"File not found: {data_path}")
            data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep=';')
            
        print(f"Loaded data with shape: {data.shape}")
        
        # Preprocessing
        # Check for missing values
        missing_values = data.isnull().sum()
        print(f"Missing values per column:\n{missing_values}")
        
        # Split features and target
        X = data.drop('quality', axis=1)
        y = data['quality']
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_split, random_state=self.random_state
        )
        
        # Store data for next steps
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = list(X.columns)
        
        print(f"Data split complete. Train set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Move to feature selection step
        self.next(self.feature_selection)

    @step
    def feature_selection(self):
        """
        Perform feature selection to identify important features
        """
        from sklearn.feature_selection import SelectKBest, f_regression
        
        print("Performing feature selection...")
        
        # Use f_regression for feature selection
        selector = SelectKBest(f_regression, k='all')
        selector.fit(self.X_train, self.y_train)
        
        # Get feature scores
        feature_scores = pd.DataFrame({
            'Feature': self.feature_names,
            'Score': selector.scores_
        })
        
        # Sort features by importance
        feature_scores = feature_scores.sort_values('Score', ascending=False)
        
        print("Top features by importance:")
        print(feature_scores.head())
        
        # Select top 8 features
        top_features = feature_scores.head(8)['Feature'].tolist()
        
        # Store selected features
        self.selected_features = top_features
        
        # Filter data to include only selected features
        self.X_train_selected = self.X_train[top_features]
        self.X_test_selected = self.X_test[top_features]
        
        print(f"Selected {len(top_features)} features: {top_features}")
        
        # Move to parallel model training
        self.next(self.train_ridge, self.train_rf, self.train_gb)
    
    @step
    def train_ridge(self):
        """
        Train and tune a Ridge regression model
        """
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import GridSearchCV
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        import time
        
        print("Training Ridge regression model...")
        start_time = time.time()
        
        # Create a pipeline with scaling and Ridge regression
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', Ridge())
        ])
        
        # Define parameter grid
        param_grid = {
            'model__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
        }
        
        # Perform grid search
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=self.cv_folds, 
            scoring='neg_mean_squared_error', n_jobs=-1,
            return_train_score=True  # Add this to get training scores
        )
        
        # Fit model
        grid_search.fit(self.X_train_selected, self.y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        # Get CV results
        cv_results = grid_search.cv_results_
        
        # Make predictions
        y_pred = best_model.predict(self.X_test_selected)
        
        # Calculate metrics
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        # Calculate training time
        training_time = time.time() - start_time
        
        # Store results
        self.model = best_model
        self.model_type = 'Ridge'
        self.best_params = best_params
        self.metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'training_time': training_time
        }
        
        # Store CV results
        self.cv_results = {
            'mean_test_score': np.abs(cv_results['mean_test_score']),  # Convert negative MSE to positive
            'std_test_score': cv_results['std_test_score'],
            'mean_train_score': np.abs(cv_results['mean_train_score']),  # Convert negative MSE to positive
            'params': cv_results['params']
        }
        
        print(f"Ridge model training complete in {training_time:.2f} seconds.")
        print(f"Best parameters: {best_params}")
        print(f"RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        # Move to join step
        self.next(self.join_models)
    
    @step
    def train_rf(self):
        """
        Train and tune a Random Forest model
        """
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        import time
        
        print("Training Random Forest model...")
        start_time = time.time()
        
        # Create Random Forest model
        rf = RandomForestRegressor(random_state=self.random_state)
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
        
        # Perform grid search
        grid_search = GridSearchCV(
            rf, param_grid, cv=self.cv_folds, 
            scoring='neg_mean_squared_error', n_jobs=-1,
            return_train_score=True
        )
        
        # Fit model
        grid_search.fit(self.X_train_selected, self.y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        # Get CV results
        cv_results = grid_search.cv_results_
        
        # Make predictions
        y_pred = best_model.predict(self.X_test_selected)
        
        # Calculate metrics
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        # Calculate feature importances
        feature_importances = best_model.feature_importances_
        
        # Calculate training time
        training_time = time.time() - start_time
        
        # Store results
        self.model = best_model
        self.model_type = 'RandomForest'
        self.best_params = best_params
        self.metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'training_time': training_time
        }
        
        # Store CV results
        self.cv_results = {
            'mean_test_score': np.abs(cv_results['mean_test_score']),
            'std_test_score': cv_results['std_test_score'],
            'mean_train_score': np.abs(cv_results['mean_train_score']),
            'params': cv_results['params']
        }
        
        # Store feature importances
        self.feature_importances = dict(zip(self.selected_features, feature_importances))
        
        print(f"Random Forest model training complete in {training_time:.2f} seconds.")
        print(f"Best parameters: {best_params}")
        print(f"RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        # Move to join step
        self.next(self.join_models)
    
    @step
    def train_gb(self):
        """
        Train and tune a Gradient Boosting model
        """
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        import time
        
        print("Training Gradient Boosting model...")
        start_time = time.time()
        
        # Create Gradient Boosting model
        gb = GradientBoostingRegressor(random_state=self.random_state)
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
        
        # Perform grid search
        grid_search = GridSearchCV(
            gb, param_grid, cv=self.cv_folds, 
            scoring='neg_mean_squared_error', n_jobs=-1,
            return_train_score=True
        )
        
        # Fit model
        grid_search.fit(self.X_train_selected, self.y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        # Get CV results
        cv_results = grid_search.cv_results_
        
        # Make predictions
        y_pred = best_model.predict(self.X_test_selected)
        
        # Calculate metrics
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        # Calculate feature importances
        feature_importances = best_model.feature_importances_
        
        # Calculate training time
        training_time = time.time() - start_time
        
        # Store results
        self.model = best_model
        self.model_type = 'GradientBoosting'
        self.best_params = best_params
        self.metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'training_time': training_time
        }
        
        # Store CV results
        self.cv_results = {
            'mean_test_score': np.abs(cv_results['mean_test_score']),
            'std_test_score': cv_results['std_test_score'],
            'mean_train_score': np.abs(cv_results['mean_train_score']),
            'params': cv_results['params']
        }
        
        # Store feature importances
        self.feature_importances = dict(zip(self.selected_features, feature_importances))
        
        print(f"Gradient Boosting model training complete in {training_time:.2f} seconds.")
        print(f"Best parameters: {best_params}")
        print(f"RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        # Move to join step
        self.next(self.join_models)
    
    @step
    def join_models(self, inputs):
        """
        Join step to collect results from all model training steps
        """
        print("Collecting results from all models...")
        
        # Create list of model results
        self.models = [
            (inp.model, inp.model_type, inp.metrics, inp.best_params, 
             getattr(inp, 'feature_importances', None), inp.cv_results)
            for inp in inputs
        ]
        
        # Sort models by R² score (higher is better)
        sorted_models = sorted(self.models, key=lambda x: -x[2]['r2'])
        
        # Select best model
        best_model, best_model_type, best_metrics, best_params, best_importances, best_cv = sorted_models[0]
        
        # Store the best model and its information
        self.best_model = best_model
        self.best_model_type = best_model_type
        self.best_metrics = best_metrics
        self.best_params = best_params
        self.best_importances = best_importances
        self.best_cv_results = best_cv
        
        # Important: We need to carry over training and test data as well
        self.X_train = inputs[0].X_train
        self.X_test = inputs[0].X_test
        self.y_train = inputs[0].y_train
        self.y_test = inputs[0].y_test
        self.feature_names = inputs[0].feature_names
        self.selected_features = inputs[0].selected_features
        self.X_train_selected = inputs[0].X_train_selected
        self.X_test_selected = inputs[0].X_test_selected
        
        print(f"Model comparison complete.")
        print(f"Best model: {best_model_type}")
        print(f"Best model metrics: {best_metrics}")
        
        # Now move to the log step
        self.next(self.log_to_mlflow)
    
    @step
    def log_to_mlflow(self):
        """
        Log all models and results to MLflow
        """
        import mlflow
        import mlflow.sklearn
        import time
        
        # Set up MLflow tracking
        mlflow.set_tracking_uri('http://localhost:5001')
        mlflow.set_experiment('wine-quality-experiment')
        
        print("Logging all models to MLflow...")
        
        # Log all models with detailed information
        for model, model_type, metrics, params, importances, cv_results in self.models:
            with mlflow.start_run():
                # Log model type and run metadata
                mlflow.log_param('model_type', model_type)
                mlflow.set_tag('execution_date', time.strftime('%Y-%m-%d %H:%M:%S'))
                mlflow.set_tag('metaflow_run_id', current.run_id)
                
                # Log dataset information
                mlflow.log_param('dataset_size', len(self.X_train) + len(self.X_test))
                mlflow.log_param('test_split_ratio', self.test_split)
                mlflow.log_param('random_state', self.random_state)
                mlflow.log_param('cv_folds', self.cv_folds)
                
                # Log selected features
                for i, feature in enumerate(self.selected_features):
                    mlflow.log_param(f'selected_feature_{i+1}', feature)
                
                # Log hyperparameters
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)
                
                # Log performance metrics
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                
                # Log cross-validation results
                for i, mean_score in enumerate(cv_results['mean_test_score']):
                    mlflow.log_metric(f'cv_mean_test_mse_{i}', mean_score)
                    mlflow.log_metric(f'cv_mean_train_mse_{i}', cv_results['mean_train_score'][i])
                    mlflow.log_metric(f'cv_std_test_mse_{i}', cv_results['std_test_score'][i])
                
                # Log feature importances if available
                if importances:
                    for feature_name, importance in importances.items():
                        mlflow.log_metric(f'importance_{feature_name}', importance)
                
                # Tag as candidate model
                mlflow.set_tag('candidate_model', 'true')
                
                # If this is the best model, tag it and register it
                if model_type == self.best_model_type:
                    mlflow.set_tag('best_model', 'true')
                    mlflow.sklearn.log_model(
                        model,
                        artifact_path="model",
                        registered_model_name="wine-quality-model"
                    )
                else:
                    # Just log the model without registering
                    mlflow.sklearn.log_model(model, artifact_path="model")
        
        print("All models successfully logged to MLflow")
        print(f"Best model ({self.best_model_type}) registered as 'wine-quality-model'")
        
        # Move to the final step
        self.next(self.end)
    
    @step
    def end(self):
        """
        Final step to summarize results
        """
        print("\n========== Training Flow Complete ==========")
        print(f"Best model: {self.best_model_type}")
        print(f"Best parameters: {self.best_params}")
        print(f"Metrics:")
        for metric, value in self.best_metrics.items():
            if metric != 'training_time':  # Skip training time in the summary
                print(f"  {metric}: {value:.4f}")
        print(f"Training time: {self.best_metrics['training_time']:.2f} seconds")
        print(f"Selected features: {self.selected_features}")
        
        if self.best_importances:
            print("\nFeature importances:")
            sorted_importances = sorted(self.best_importances.items(), key=lambda x: -x[1])
            for feature, importance in sorted_importances:
                print(f"  {feature}: {importance:.4f}")
        
        print("\nModel has been registered in MLflow as 'wine-quality-model'")
        print("You can now use the scoring flow to make predictions.")

if __name__ == '__main__':
    WineQualityTrainingFlow()