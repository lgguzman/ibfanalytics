from itertools import combinations

from pyspark.ml.fpm import FPGrowth
from pyspark.sql.functions import col, size
from sbxpy import SbxCore
from functools import reduce
import asyncio
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import mean as _mean, stddev as _stddev, col, to_json, from_json, struct, lit
from pyspark.sql import functions as func
from pyspark.sql.types import StructType, StructField
from pyspark.ml.linalg import VectorUDT
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans, BisectingKMeans, GaussianMixture
import matplotlib.pyplot as plt
from pyspark.ml.feature import PCA as PCAml


class AssociationRules:

    async def df_sbx_cart_item(self, sbx, spark, min_transaction=6, attribute='variety_name', customer=None, to_suggested=False):
        query = sbx.with_model('cart_box_item').fetch_models(['variety', 'cart_box', 'product_group']) \
            .and_where_is_not_null('cart_box.purchase')
        if customer  is not None:
            query = query.and_where_in('cart_box.customer', customer)

        if not to_suggested:
            total_data = await query.find_all_query()
        else:
            query.set_page(0)
            query.set_page_size(5)
            temp = await query.find()
            total_data = [temp]
        d = {}
        errors = []
        for data in total_data:
            for item in data['results']:
                try:
                    # item['name'] = data['fetched_results']['add_masterlist'][
                    #     data['fetched_results']['inventory'][item['inventory']]['masterlist']][attribute]
                    variety = data['fetched_results']['variety'][item['variety']]['variety_name']
                    product_group = data['fetched_results']['product_group'][item['product_group']]['common_name']
                    item['name'] = product_group #+ ' ' + variety
                    if data['fetched_results']['cart_box'][item['cart_box']]['purchase'] not in d:
                        d[data['fetched_results']['cart_box'][item['cart_box']]['purchase']] = {}
                    d[data['fetched_results']['cart_box'][item['cart_box']]['purchase']][item['name']] = 1
                except Exception as inst:
                    errors.append(inst)
        print("Errors")
        print(errors)
        if not to_suggested:
            varieties = [(key, [k for k, val in value.items()]) for key, value in d.items()]
            varieties = list(filter(lambda t: len(t[1]) > min_transaction, varieties))
            return spark.createDataFrame(varieties, ["id", "items"]).repartition(100)
        else:
            def merging_data(varieties, elements):
                for element in elements:
                    if element not in varieties:
                        varieties.append(element)
                return varieties

            varieties = [[k for k, val in value.items()] for key, value in d.items()]
            varieties = reduce(merging_data, varieties, [])
            return await self.transform_possible_list(spark, varieties)


    async def get_model(self, df, min_support=0.1, min_confidence=0.6):
        fpGrowth = FPGrowth(itemsCol="items", minSupport=min_support, minConfidence=min_confidence)
        model = fpGrowth.fit(df)
        return model.freqItemsets.sort("freq", ascending=False), model.associationRules.sort("confidence", ascending=True), model


    async def transform_possible_list(self, spark, items):
        combins = sorted(combinations(items), key=lambda comb: len(comb), reverse=True)
        possibles = [(str(i), combins[i])  for i in range(len(combins))]
        return spark.createDataFrame(possibles, ["id", "items"]).repartition(100)



    async def run_suggested(self, model, df):
        return model.transform(df).withColumn("item_size", size(col("items"))) \
            .withColumn("prediction_size", size(col("prediction"))) \
            .orderBy(["item_size", "prediction"], ascending=[1, 0])


    async def  test(self):
        sbx = SbxCore()
        sbx.initialize(os.environ['DOMAIN'], os.environ['APP-KEY'], os.environ['SERVER_URL'])
        login = await sbx.login(os.environ['LOGIN'], os.environ['PASSWORD'], os.environ['DOMAIN'] )


        spark = SparkSession.builder.appName("SparkExample").getOrCreate()

        df = await self.df_sbx_cart_item(sbx, spark, 6, '_KEY')
        freq_items, association_rules, model  = await self.get_model(df, 0.1, 0.6)

        suggested = await self.run_suggested(model, df)
        suggested.drop('items').show(10, False)

        spark.stop()

    def run_test(self):
        loop = asyncio.new_event_loop()
        loop.run_until_complete(self.test())




class UserCluster:


    async def df_sbx_customer_purchase_behaivor(self, sbx, spark, date_int, limit =3, variety_sw = False):
        data_complete = await  sbx.with_model('cart_box_item').fetch_models(['cart_box']) \
            .and_where_greater_than('cart_box.charge_date', date_int) \
            .and_where_is_not_null('cart_box.purchase').find_all_query()
        sc = spark.sparkContext
        d = {}
        groups = {}
        errors = []

        for data in  data_complete:
            for item in data['results']:
                try:
                    customer = data['fetched_results']['cart_box'][item['cart_box']]['customer']
                    if customer not in d:
                        d[customer] = {}
                    purchase = item['product_group'] + ((' ' + item['variety']) if variety_sw else '')
                    if purchase not in d[customer]:
                        d[customer][purchase] = 1
                    # else:
                    #     d[customer][purchase] =  d[customer][purchase] + 1
                    if purchase not in groups:
                        groups[purchase] = 1
                except Exception as inst:
                    errors.append(inst)

        columns = [key for key, value in groups.items()]
        def newRow(data, customer, columns):
            row = {}
            row['customer'] = customer
            row['products'] = []
            for key  in columns:
                if key not in data:
                    row['products'].append(0)
                else:
                    row['products'].append(data[key])
            return row

        customers = [ newRow(value, key, columns) for key, value in d.items()]
        customers = list(filter(lambda x: sum(x['products']) > limit, customers) )

        tmp = sc.parallelize(customers, numSlices=100)
        df = spark.read.option("multiLine", "true").json(tmp)
        json_vec = to_json(struct(struct(
            lit(1).alias("type"),  # type 1 is dense, type 0 is sparse
            col("products").alias("values")
        ).alias("v")))

        schema = StructType([StructField("v", VectorUDT())])

        return df.withColumn(
            "features", from_json(json_vec, schema).getItem("v")
        )


    async def df_sbx_customer_special_box_purchased(self, sbx, spark):
        data = await  sbx.with_model('cart_box') \
            .set_page_size(1000) \
            .and_where_is_not_null('purchase') \
            .and_where_is_equal('variety', os.environ['SPECIAL_BOX']).find()
        sc = spark.sparkContext
        def deleteMeta(d):
            dt = {}
            dt['customer'] = d['customer']
            dt['total_items'] = d['total_items']
            dt['current_percentage'] = d['current_percentage']
            dt['count'] = 1
            return dt

        dit = list(map(deleteMeta, data['results']))
        tmp = sc.parallelize(dit, numSlices=100)
        df = spark.read.option("multiLine", "true").json(tmp)
        df2 = df.groupBy("customer").agg(func.avg("total_items").alias('total_items'),
                                          func.avg("current_percentage").alias('current_percentage'),
                                          func.sum("count").alias('count'))

        (cumean, custd, comean, costd, tmean, tstd) = df2.select(
            _mean(col('current_percentage')).alias('cumean'),
            _stddev(col('current_percentage')).alias('custd'),
            _mean(col('count')).alias('comean'),
            _stddev(col('count')).alias('costd'),
            _mean(col('total_items')).alias('total_items'),
            _stddev(col('total_items')).alias('total_items'),
        ).first()
        df3 = df2.withColumn("acurrent_percentage", (col("current_percentage") - cumean) / custd).withColumn("acount", (
                    col("count") - comean) / costd).withColumn("atotal_items", (col("total_items") - tmean) / tstd)
        vecAssembler = VectorAssembler(inputCols=["acurrent_percentage", "acount", "atotal_items"],
                                       outputCol="features")
        return vecAssembler.transform(df3)

    async def k_means(self,df, k, seed):
        if k is not None and seed is not None:
            kmeans = KMeans(k=k, seed=seed)
        else:
            kmeans = KMeans()
        return  kmeans.fit(df.select('features'))

    async def bisecting_means(self,df):
        return BisectingKMeans().fit(df.select('features'))

    async def gaussian_mixture(self,df):
        return GaussianMixture().fit(df.select('features'))

    async def run_cluster(self,model, df):
        return model.transform(df)

    async def plot_cluster(self, df, x='_3', y= '_4'):
        pca = PCAml(k=2, inputCol="features", outputCol="pca")
        model3 = pca.fit(df)
        transformed2 = model3.transform(df)

        def extract(row):
            return (row.customer,) + (row.prediction,) + tuple(row.pca.toArray().tolist())

        pcadf = transformed2.rdd.map(extract).toDF(["customer", "prediction"])
        pcadf.show(10, False)
        pandad = pcadf.toPandas()
        pandad.plot.scatter(x=x, y=y, c='prediction', colormap='viridis')
        plt.show()

    async def test(self):
        sbx = SbxCore()
        sbx.initialize(os.environ['DOMAIN'], os.environ['APP-KEY'], os.environ['SERVER_URL'])
        login = await sbx.login(os.environ['LOGIN'], os.environ['PASSWORD'], os.environ['DOMAIN2'] )

        spark = SparkSession.builder.appName("SparkExample").getOrCreate()

        df = await self.df_sbx_customer_special_box_purchased(sbx,spark)
        df.show()

        model = await self.k_means(df,3,1)
        transformed = await self.run_cluster(model, df)

        # transformed.select(['customer','current_percentage', 'count' , 'total_items', 'prediction']).show(20, False)

        await self.plot_cluster(transformed)
        spark.stop()

    def run_test(self):
        loop = asyncio.new_event_loop()
        loop.run_until_complete(self.test())
