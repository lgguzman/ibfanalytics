import os
import sys


async def cluster_by_purchase_behaivor(user_cluster, association):
    sbx = SbxCore()
    sbx.initialize(os.environ['DOMAIN'], os.environ['APP-KEY'], 'https://sbxcloud.com/api')
    login = await sbx.login(os.environ['LOGIN'], os.environ['PASSWORD'], os.environ['DOMAIN'])

    spark = SparkSession.builder.appName("SparkExample").getOrCreate()
    print('fetch data')
    df = await user_cluster.df_sbx_customer_purchase_behaivor(sbx, spark, os.environ['DATE_INT'])

    total = df.count()
    best_score = total*100
    best_df = None
    clustering_types=[user_cluster.bisecting_means ,user_cluster.k_means]
    total_clustring_types = len(clustering_types)
    k=3
    print('get_bestCluster')
    for i in range(total_clustring_types):
        if i == total_clustring_types-1:
            model = await clustering_types[i](df, k, 1)
        else:
            model = await clustering_types[i](df)
        transformed = await user_cluster.run_cluster(model, df)
        grouped =  transformed.groupBy("prediction").agg(countDistinct('customer').alias("count"))
        (std) = grouped.select(
            _stddev(col('count')).alias('std')
        ).first()
        grouped.show()
        score = std['std']
        if (score < best_score):
            best_score = score
            best_df = transformed
        k = len(model.clusterCenters())
        print('with ' + str(k) + ' clusters')


    # await user_cluster.plot_cluster(best_df)
    grouped =  best_df.groupBy("prediction").agg(collect_list(col("customer")).alias("customers"),countDistinct('customer').alias("count"))\
        .sort("count", ascending=False)
    print('Best Clusters')
    grouped.show()
    bigger_cluster = grouped.first()
    print('Fetch data to Associate')
    df = await association.df_sbx_cart_item(sbx, spark, 3, customer=bigger_cluster['customers'])
    freq_items, association_rules, model = await association.get_model(df, 0.01, 0.1)

    suggested = await association.run_suggested(model, df)
    suggested.drop('items').show(10, False)
    spark.stop()

if __name__ == "__main__":
    if os.path.exists('ibfanalytics.zip'):
        sys.path.insert(0, 'ibfanalytics.zip')
    else:
        sys.path.insert(0, './ibfanalytics')

    from ibfanalytics import UserCluster
    from ibfanalytics import AssociationRules
    from sbxpy import SbxCore
    import asyncio
    from pyspark.sql import SparkSession
    import argparse
    from pyspark.sql.functions import countDistinct, col, stddev as _stddev, collect_list

    parser = argparse.ArgumentParser()
    parser.add_argument('--user_name',type=str, required=True, help='user name')
    parser.add_argument('--password', type=str, required=True, help='password')
    parser.add_argument('--app_key', type=str, required=True, default=None, help="app key")
    parser.add_argument('--domain', type=str, required=True, default=None, help="domain")
    parser.add_argument('--cluster_year', type=str, required=True, default=None, help="cluster year")
    args = parser.parse_args()
    os.environ['DOMAIN'] = args.domain
    os.environ['APP-KEY'] = args.app_key
    os.environ['LOGIN'] = args.user_name
    os.environ['PASSWORD'] = args.password
    os.environ['DATE_INT'] = args.cluster_year

    user = UserCluster()
    association = AssociationRules()
    loop = asyncio.new_event_loop()
    loop.run_until_complete(cluster_by_purchase_behaivor(user, association))