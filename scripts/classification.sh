for model in catboost lightgbm xgboost tabnet
do
    python src/run_classification.py \
    models=$model \
    log.experiment=False
done