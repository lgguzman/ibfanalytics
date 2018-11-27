import os
import sys


async def cluster_by_purchase_behaivor(sbx, spark, user_cluster, date_int):

    #login = await sbx.login(os.environ['LOGIN'], os.environ['PASSWORD'], os.environ['DOMAIN'])
    print('fetch data')
    df = await user_cluster.df_sbx_customer_purchase_behaivor(sbx, spark, date_int)
    total = df.count()
    best_score = total*100
    best_df = None
    print('Creating clusters')
    clustering_types=[user_cluster.bisecting_means ,user_cluster.k_means]
    total_clustring_types = len(clustering_types)
    k=3
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
        #grouped.show()
        score = std['std']
        #await user_cluster.plot_cluster(transformed)
        if (score < best_score):
            best_score = score
            best_df = transformed
        k = len(model.clusterCenters())



    return best_df.groupBy("prediction").agg(collect_list(col("customer")).alias("customers"),countDistinct('customer').alias("count"))\
        .sort("count", ascending=False)


async def association_rule_by_customer_set(sbx, spark, association,  customers):
    df = await association.df_sbx_cart_item(sbx, spark, 2, customer=customers, attribute='variety')
    freq_items, association_rules, model = await association.get_model(df, 0.03, 0.6)
    # # Display frequent itemsets.
    #freq_items.withColumn("data_size", size(col("items")) / col("freq")).sort("data_size", ascending=False).show(20,
    #                                                                                                               False)
    #
    # # Display generated association rules.
    # association_rules.sort("confidence", ascending=True).show(500, False)

    return model

    # suggested = await association.run_suggested(model, df)
    # suggested.drop('items').show(10, False)


    # model.transform(df).withColumn("item_size", size(col("items"))) \
    #     .withColumn("prediction_size", size(col("prediction"))) \
    #     .orderBy(["item_size", "prediction"], ascending=[1, 0]).drop('items').show(10, False)


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
    parser.add_argument('--app_key2', type=str, required=True, default=None, help="app key2")
    parser.add_argument('--domain2', type=str, required=True, default=None, help="domain2")
    parser.add_argument('--domain', type=str, required=True, default=None, help="domain")
    parser.add_argument('--customer', type=str, required=True, default=None, help="customer")
    parser.add_argument('--cluster_year', type=str, required=True, default=None, help="cluster year")
    args = parser.parse_args()
    domain = args.domain
    app_key = args.app_key
    token = args.token
    customer = args.customer
    date_int = args.cluster_year

    user = UserCluster()
    association = AssociationRules()
    loop = asyncio.new_event_loop()
    sbx = SbxCore()
    sbx.initialize(domain, app_key, 'https://sbxcloud.com/api')
    sbx.headers['Authorization'] = "Bearer " + token
    spark = SparkSession.builder.appName("Suggested").getOrCreate()
    grouped =  loop.run_until_complete(cluster_by_purchase_behaivor(sbx, spark, user, date_int))
    tasks = [association.df_sbx_cart_item(sbx, spark, 1, customer=[customer], attribute='variety', to_suggested=True)]
    print('Creating Association Rules')
    for row in grouped.collect():
        tasks.append(association_rule_by_customer_set(sbx,spark,association,row['customers']))

    futures = [asyncio.ensure_future(t, loop=loop) for t in tasks]
    gathered = asyncio.gather(*futures, loop=loop, return_exceptions=True)
    models =  loop.run_until_complete(gathered)
    has_exception=False
    for data in models:
        if isinstance(data, Exception):
            print(data)
            has_exception=True
    if not has_exception:
        product_list = models[0]
        df =  models[1].transform(product_list)
        for i in range(len(models)-2):
            df = df.union(models[i+2].transform(product_list))
        df = df.dropDuplicates(["items"]).withColumn("item_size", size(col("items"))) \
                .withColumn("prediction_size", size(col("prediction"))) \
            .orderBy(["item_size", "prediction"], ascending=[1, 0]).drop('items').show(10, False)
        response = [ data for item in df.select("items").flatMap(lambda x: x).collect()  for data in  item]

        print(response)



    spark.stop()