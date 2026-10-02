"""Optional Spark classifier backend; requires an installed Spark runtime."""
import argparse
from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--csv',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    spark=SparkSession.builder.appName('ReconstructedDDoSClassifier').getOrCreate()
    features=['src_port','dest_port','packet_length','packets_per_time']
    frame=spark.read.option('header',True).csv(args.csv)
    for col in features+['target']:
        frame=frame.withColumn(col,F.col(col).cast('double'))
    frame=frame.filter(F.col('target').isin(0.0,1.0)).dropDuplicates()
    for col in features:
        valid=F.col(col)>=0
        if col in ['src_port','dest_port']:valid=valid & (F.col(col)<=65535)
        frame=frame.withColumn(col,F.when(valid,F.col(col)).otherwise(None))
    train,test=frame.randomSplit([.75,.25],seed=42)
    if train.select('target').distinct().count()!=2 or test.count()==0:
        raise ValueError('Split needs both train classes and a nonempty test set')
    clean=[col+'_imputed' for col in features]
    pipeline=Pipeline(stages=[Imputer(inputCols=features,outputCols=clean,strategy='median'),
        VectorAssembler(inputCols=clean,outputCol='features'),
        RandomForestClassifier(labelCol='target',featuresCol='features',numTrees=100,seed=42)])
    model=pipeline.fit(train);predictions=model.transform(test)
    accuracy=MulticlassClassificationEvaluator(labelCol='target',metricName='accuracy').evaluate(predictions)
    print(f'Holdout accuracy: {accuracy}')
    model.write().save(args.output+'/model')
    predictions.select(*features,'target','prediction').write.option('header',True).csv(args.output+'/predictions')
    spark.stop()
