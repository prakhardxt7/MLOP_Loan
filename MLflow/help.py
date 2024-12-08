import mlflow
mlflow.set_tracking_uri('http://localhost:5000')

exp_id = mlflow.create_experiment('Loan_Prediction')

with mlflow.start_run(run_name='RandomForest') as run:
    pass


mlflow.end_run()  #End the current active run


n_estimator=10
criterion='gini'

#logs a single key value param in currently active run
mlflow.log_param('n_estimator',n_estimator)
mlflow.log_param('criterion',criterion)


#multiple params
params = {'n_estimator':n_estimator,
          'criterion':criterion
          }

mlflow.log_params(params)


#Metric
mlflow.log_metric('accuracy',0.9)

#Setting Tag
mlflow.set_tag()

with mlflow.start_run(run_name='RandomForest') as run:
    mlflow.set_tag('version','1.0.0')
    

# logging the artifacts
mlflow.log_artifacts()