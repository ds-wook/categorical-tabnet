for model in catboost lightgbm xgboost tabnet catabnet
do
    python src/train.py \
    models=$model \
    log.experiment=False \
    data=shrutime \
    features=shrutime
done