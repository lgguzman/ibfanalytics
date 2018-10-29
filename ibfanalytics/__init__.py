from ibfanalytics.user_cluster import UserCluster
from sbxpy import SbxCore
import asyncio
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import countDistinct, col, stddev as _stddev

async def cluster_by_purchase_behaivor(user_cluster):
    sbx = SbxCore()
    sbx.initialize(os.environ['DOMAIN'], os.environ['APP-KEY'], os.environ['SERVER_URL'])
    login = await sbx.login(os.environ['LOGIN'], os.environ['PASSWORD'], os.environ['DOMAIN2'])

    spark = SparkSession.builder.appName("SparkExample").getOrCreate()

    df = await user_cluster.df_sbx_customer_purchase_behaivor(sbx, spark, os.environ['DATE_INT'])

    k=3
    total = df.count()
    best_df = None
    best_score = total*10
    while(k<10):
        model = await user_cluster.k_means(df, k, 1)
        transformed = await user_cluster.run_cluster(model, df)
        grouped = transformed.groupBy("prediction").agg(countDistinct('customer').alias("count"))
        (std) = grouped.select(
            _stddev(col('count')).alias('std')
        ).first()

        score = std['std']*k
        grouped.show()
        print(score)
        if(score < best_score):
            best_score = score
            best_df = transformed
        k=k+1

    await user_cluster.plot_cluster(best_df)
    spark.stop()


user = UserCluster()
loop = asyncio.new_event_loop()
loop.run_until_complete(cluster_by_purchase_behaivor(user))

