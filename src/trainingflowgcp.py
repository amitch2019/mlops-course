
from metaflow import FlowSpec, step, conda_base, kubernetes, resources, retry, timeout, catch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import mlflow
import numpy as np

@conda_base(libraries={
    'scikit-learn': '1.2.2',
    'mlflow': '2.10.0',
    'databricks-cli': '0.17.6'  
}, python='3.9.16')
class WineQualityTrainingFlow(FlowSpec):
    
    @step
    def start(self):
        """
        Load data and prepare for model training
        """
        from sklearn.datasets import load_wine
        
        print("Loading wine dataset...")
        wine_data = load_wine()
        X = wine_data.data
        y = wine_data.target
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2)
        
        print(f"Data loaded and split: {self.X_train.shape[0]} training samples, {self.X_test.shape[0]} test samples")
        self.next(self.train_model)
    
    @kubernetes
    @resources(cpu=1, memory=4096)  
    @retry(times=2)
    @timeout(seconds=300)  
    @catch(var="error")
    @step
    def train_model(self):
        """
        Train the model and log to MLFlow
        """
        if hasattr(self, 'error'):
            print(f"Caught an error: {self.error}")
            self.next(self.end)
            return
            
        try:
            # Set up MLFlow tracking
            print("Setting up MLFlow tracking...")
            # mlflow.set_tracking_uri("http://mlflow-server.default.svc.cluster.local:8080")
            mlflow.set_tracking_uri("http://mlflow.default.svc.cluster.local:5000")

            mlflow.set_experiment("wine-quality")
            
            # Train model
            print("Training model...")
            with mlflow.start_run(run_name="wine-quality-logistic") as run:
                # Use a simple model to reduce memory usage
                model = LogisticRegression(max_iter=500, solver='liblinear')
                model.fit(self.X_train, self.y_train)
                
                # Evaluate
                acc = model.score(self.X_test, self.y_test)
                
                # Log metrics and model
                print(f"Logging metrics to MLFlow, accuracy: {acc}")
                mlflow.log_param("solver", "liblinear")
                mlflow.log_param("max_iter", 500)
                mlflow.log_metric("accuracy", acc)
                
                # Save model
                mlflow.sklearn.log_model(model, "model")
                mlflow.register_model(f"runs:/{run.info.run_id}/model", "Wine-Quality-Model")
                
                self.accuracy = acc
                print(f"Model trained with accuracy: {acc}")
                
        except Exception as e:
            print(f"Error in train_model step: {str(e)}")
            self.accuracy = None
            
        self.next(self.end)
    
    @step
    def end(self):
        """
        End the flow
        """
        if hasattr(self, 'accuracy') and self.accuracy is not None:
            print(f"Flow completed successfully with accuracy: {self.accuracy}")
        else:
            print("Flow completed with errors in model training.")

if __name__ == "__main__":
    WineQualityTrainingFlow()