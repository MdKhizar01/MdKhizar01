# DDoS Traffic Pipeline — reconstructed local version

Clean numeric network traffic features, train a random forest, flag unusual traffic with Isolation Forest, and export predictions and evaluation metrics. Train/test splitting occurs before imputation and model fitting.

```sh
python3 -m pip install -r requirements.txt
python3 pipeline.py --demo
python3 pipeline.py --csv /path/to/traffic.csv
```

Required CSV columns: `src_port`, `dest_port`, `packet_length`, `packets_per_time`, `target` (0 or 1). Optional metadata columns are preserved during cleaning. Metrics and model files are written under `artifacts/`.

The demo generates synthetic traffic and does not establish real attack detection performance. For actual traffic, use a time-based or capture-group holdout to avoid correlated packets crossing the train/test boundary; the included random split is a baseline.

## Optional Spark and Cassandra adapters

`cassandra_schema.cql` supplies the remembered storage schema. Optional adapters require infrastructure you configure separately:

```sh
python3 -m pip install -r optional-requirements.txt
# First initialize Cassandra using cassandra_schema.cql.
python3 load_cassandra.py --csv /path/to/clean-traffic.csv
spark-submit spark_pipeline.py --csv /path/or/hdfs/traffic.csv --output /new/output/location
```

The Cassandra loader expects clean integer port/length/target fields; it inserts new UUIDs each run and is not idempotent. The Spark adapter trains directly from CSV and refuses to overwrite an existing output. Neither adapter was exercised against live infrastructure in the reconstruction environment.

This reconstruction does not implement or claim a tested Hadoop/Hive/SageMaker deployment, nor an integrated Cassandra-to-Spark data path. See the existing Hadoop report linked from the portfolio.
