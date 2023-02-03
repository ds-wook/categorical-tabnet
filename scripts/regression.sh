for model in catboost lightgbm xgboost tabnet
do
    python src/run_regression.py \
    models=$model \
    log.experiment=False
done