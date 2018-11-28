import os
import sys
import time

async def cluster_by_purchase_behaivor(sbx, spark, user_cluster, date_int,kmeans=3):

    print('fetch data')
    df = await user_cluster.df_sbx_customer_purchase_behaivor(sbx, spark, date_int)
    total = df.count()
    best_score = total*100
    best_df = None
    print('Creating clusters')
    clustering_types=[user_cluster.bisecting_means ,user_cluster.k_means]
    total_clustring_types = len(clustering_types)
    k=kmeans
    for i in range(total_clustring_types):
        if i == total_clustring_types-1:
            model = await clustering_types[i](df, k, 1)
        else:
            model = await clustering_types[i](df,k)
        transformed = await user_cluster.run_cluster(model, df)
        grouped = transformed.groupBy("prediction").agg(countDistinct('customer').alias("count"))
        (std) = grouped.select(
            _stddev(col('count')).alias('std')
        ).first()
        score = std['std']
        if (score < best_score):
            best_score = score
            best_df = transformed
        k = len(model.clusterCenters())



    return best_df.groupBy("prediction").agg(collect_list(col("customer")).alias("customers"),countDistinct('customer').alias("count"))\
        .sort("count", ascending=False)


async def association_rule_by_customer_set(sbx, spark, association,  customers):
    df = await association.df_sbx_cart_item(sbx, spark, 4, customer=customers, attribute='variety')
    freq_items, association_rules, model = await association.get_model(df, 0.02, 0.3)
    # # Display frequent itemsets.
    #freq_items.withColumn("data_size", size(col("items")) / col("freq")).sort("data_size", ascending=False).show(20,
    #                                                                                                               False)
    # # Display generated association rules.
    #association_rules.sort("confidence", ascending=True).show(500, False)
    return model


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
    from pyspark.sql.functions import countDistinct, col, stddev as _stddev, collect_list, size

    parser = argparse.ArgumentParser()
    parser.add_argument('--token',type=str, required=True, help='token')
    parser.add_argument('--app_key', type=str, required=True, default=None, help="app key")
    parser.add_argument('--app_key2', type=str, required=False, default=None, help="app key2")
    parser.add_argument('--domain2', type=str, required=False, default=None, help="domain2")
    parser.add_argument('--domain', type=str, required=True, default=None, help="domain")
    parser.add_argument('--customer', type=str, required=False, default=None, help="customer")
    parser.add_argument('--customer2', type=str, required=False, default=None, help="customer2")
    parser.add_argument('--varieties', type=str, required=False, default=None, help="varieties")
    parser.add_argument('--varieties2', type=str, required=False, default=None, help="varieties2")
    parser.add_argument('--kmeans', type=str, required=True, default=None, help="kmeans")
    parser.add_argument('--cluster_year', type=str, required=True, default=None, help="cluster year")
    args = parser.parse_args()
    domain = args.domain
    app_key = args.app_key
    token = args.token
    customer = args.customer
    date_int = args.cluster_year
    kmeans = int(args.kmeans)
    str_varieties = args.varieties

    if str_varieties is not None or customer is not None:

        user = UserCluster()
        association = AssociationRules()
        loop = asyncio.new_event_loop()
        sbx = SbxCore()
        sbx.initialize(domain, app_key, 'https://sbxcloud.com/api')
        sbx.headers['Authorization'] = "Bearer " + token
        spark = SparkSession.builder.appName("Suggested").getOrCreate()
        start_time = time.time()
        grouped = loop.run_until_complete(cluster_by_purchase_behaivor(sbx, spark, user, date_int, kmeans))
        print("--- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        task = []
        if customer is not None:
            tasks = [association.df_sbx_cart_item(sbx, spark, 1, customer=[customer], attribute='variety', to_suggested=True)]
        else:
            tasks = [association.df_from_varieties(sbx, spark, str_varieties.split(','))]
        print('Creating Association Rules')
        for row in grouped.collect():
            tasks.append(association_rule_by_customer_set(sbx, spark, association, row['customers']))

        futures = [asyncio.ensure_future(t, loop=loop) for t in tasks]
        gathered = asyncio.gather(*futures, loop=loop, return_exceptions=True)
        models =  loop.run_until_complete(gathered)
        print("--- %s seconds ---" % (time.time() - start_time))
        print("Get Suggested")
        start_time = time.time()
        has_exception=True
        succesful_models = []
        if not isinstance(models[0], Exception):
            product_list = models[0]
            for i in range(1,len(models)):
                if not isinstance(models[i], Exception):
                    succesful_models.append(models[i])
        if len(succesful_models) > 0:
            df = succesful_models[0].transform(product_list)
            for i in range(1, len(succesful_models)):
                df = df.union(succesful_models[i].transform(product_list))
            df = df.dropDuplicates(["prediction"]).withColumn("item_size", size(col("items"))) \
                .withColumn("prediction_size", size(col("prediction"))) \
                .orderBy(["item_size", "prediction"], ascending=[1, 0])
            response = list(set([data for item in df.select("prediction").rdd.flatMap(lambda x: x).collect() for data in item]))
            response = loop.run_until_complete(association.get_varieties_from_product_color(sbx,response))
            print(response)
            print("--- %s seconds ---" % (time.time() - start_time))

        spark.stop()