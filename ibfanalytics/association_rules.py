from sbxpy import SbxCore
import asyncio
import os
from pyspark.sql import SparkSession
from pyspark.ml.fpm import FPGrowth
from pyspark.sql.functions import  col, size


class AssociationRules:

    async def df_sbx_cart_item(self, sbx, spark, min_transaction=6, attribute='search_name'):
        data = await  sbx.with_model('cart_box_item').fetch_models(['inventory.masterlist', 'cart_box']) \
            .set_page_size(1000) \
            .and_where_is_not_null('cart_box.purchase').find()
        d = {}
        errors = []
        for item in data['results']:
            try:
                item['name'] = data['fetched_results']['add_masterlist'][
                    data['fetched_results']['inventory'][item['inventory']]['masterlist']][attribute]
                if data['fetched_results']['cart_box'][item['cart_box']]['purchase'] not in d:
                    d[data['fetched_results']['cart_box'][item['cart_box']]['purchase']] = {}
                d[data['fetched_results']['cart_box'][item['cart_box']]['purchase']][item['name']] = 1
            except Exception as inst:
                errors.append(inst)
        varieties = [(key, [k for k, val in value.items()]) for key, value in d.items()]
        varieties = list(filter(lambda t: len(t[1]) > min_transaction, varieties))
        return spark.createDataFrame(varieties, ["id", "items"])


    async def get_model(self, df, min_support=0.1, min_confidence=0.6):
        fpGrowth = FPGrowth(itemsCol="items", minSupport=min_support, minConfidence=min_confidence)
        model = fpGrowth.fit(df)
        return model.freqItemsets.sort("freq", ascending=False), model.associationRules.sort("confidence", ascending=True), model

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



