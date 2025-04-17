# src/scoringflow.py
from metaflow import FlowSpec, step, Parameter, Flow, JSONType
import pandas as pd
import numpy as np
import os

class WineQualityScoringFlow(FlowSpec):
    """
    A flow for scoring new data using the best model trained by WineQualityTrainingFlow
    """
    # Define parameters
    input_data = Parameter('input_data', 
                          help='Path to CSV file with wine data to score',
                          default=None)
    sample_vector = Parameter('sample_vector', 
                              type=JSONType,
                              help='Sample wine data as JSON array to score',
                              default=None)
    use_test_data = Parameter('use_test_data', 
                              help='Whether to use test data from training flow',
                              default=True,
                              type=bool)
    model_name = Parameter('model_name',
                          help='Name of the registered model to use',
                          default='wine-quality-model')
    model_version = Parameter('model_version',
                             help='Version of the model to use (latest if not specified)',
                             default=None)

    @step
    def start(self):
        """
        Start the flow by loading the model and preparing input data
        """
        import mlflow
        import mlflow.sklearn
        
        # Set up MLflow tracking
        mlflow.set_tracking_uri('http://localhost:5001')
        
        print("Loading model and preparing data...")
        
        # Get the latest training flow run
        train_run = Flow('WineQualityTrainingFlow').latest_run
        self.train_run_id = train_run.pathspec
        
        # Load the selected features to ensure we use the same features for prediction
        self.selected_features = train_run['end'].task.data.selected_features
        
        print(f"Selected features: {self.selected_features}")
        
        # Load the model from MLflow
        if self.model_version:
            model_uri = f"models:/{self.model_name}/{self.model_version}"
        else:
            model_uri = f"models:/{self.model_name}/latest"
        
        print(f"Loading model from {model_uri}")
        self.model = mlflow.sklearn.load_model(model_uri)
        self.model_type = train_run['end'].task.data.best_model_type
        
        # Determine which input data to use
        if self.sample_vector:
            print(f"Using provided sample vector: {self.sample_vector}")
            # Convert to DataFrame
            self.input_df = pd.DataFrame([self.sample_vector], columns=self.selected_features)
            self.input_type = 'sample'
            
        elif self.input_data:
            print(f"Loading input data from {self.input_data}")
            # Load data from file
            input_path = os.path.expanduser(self.input_data)
            self.input_df = pd.read_csv(input_path, sep=';')
            # Ensure we only use the selected features
            self.input_df = self.input_df[self.selected_features]
            self.input_type = 'file'
            
        elif self.use_test_data:
            print("Using test data from training flow")
            # Use test data from training flow
            self.input_df = train_run['end'].task.data.X_test_selected
            self.true_values = train_run['end'].task.data.y_test
            self.input_type = 'test'
            
        else:
            print("No input data provided!")
            self.input_type = None
            # Create an empty DataFrame with the right columns to avoid errors
            self.input_df = pd.DataFrame(columns=self.selected_features)
            
        # Unconditionally go to predict step
        self.next(self.predict)
        
    @step
    def predict(self):
        """
        Make predictions on the input data
        """
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        
        print("Making predictions...")
        
        # Check if we have valid input data
        if self.input_type is None or self.input_df.empty:
            print("No valid input data to make predictions")
            self.has_predictions = False
        else:
            # Make predictions
            self.predictions = self.model.predict(self.input_df)
            self.has_predictions = True
            
            # If we have true values, calculate metrics
            if self.input_type == 'test':
                self.mse = mean_squared_error(self.true_values, self.predictions)
                self.rmse = np.sqrt(self.mse)
                self.mae = mean_absolute_error(self.true_values, self.predictions)
                self.r2 = r2_score(self.true_values, self.predictions)
                
                print(f"Test data metrics:")
                print(f"  RMSE: {self.rmse:.4f}")
                print(f"  MAE: {self.mae:.4f}")
                print(f"  R²: {self.r2:.4f}")
        
        # Move to final step
        self.next(self.end)
    
    @step
    def end(self):
        """
        Final step to output predictions
        """
        print("\n========== Scoring Flow Complete ==========")
        
        if not hasattr(self, 'has_predictions') or not self.has_predictions:
            print("No predictions made. Please provide input data.")
            return
        
        print(f"Model used: {self.model_type}")
        
        if self.input_type == 'sample':
            # Print prediction for single sample
            print(f"Input data: {self.sample_vector}")
            print(f"Predicted wine quality: {self.predictions[0]:.2f}")
            
        elif self.input_type == 'file':
            # Print summary of predictions for file data
            print(f"Predictions made for {len(self.predictions)} samples")
            print(f"Prediction summary:")
            print(f"  Min: {min(self.predictions):.2f}")
            print(f"  Max: {max(self.predictions):.2f}")
            print(f"  Mean: {np.mean(self.predictions):.2f}")
            print(f"  Median: {np.median(self.predictions):.2f}")
            
        elif self.input_type == 'test':
            # Print metrics
            print(f"Test data metrics:")
            print(f"  RMSE: {self.rmse:.4f}")
            print(f"  MAE: {self.mae:.4f}")
            print(f"  R²: {self.r2:.4f}")
            print(f"Prediction summary for {len(self.predictions)} test samples:")
            print(f"  Min: {min(self.predictions):.2f}")
            print(f"  Max: {max(self.predictions):.2f}")
            print(f"  Mean: {np.mean(self.predictions):.2f}")
            print(f"  Median: {np.median(self.predictions):.2f}")

if __name__ == '__main__':
    WineQualityScoringFlow()