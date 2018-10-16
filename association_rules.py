from sbxpy import SbxCore
import asyncio
import os
from pyspark.sql import SparkSession
from pyspark.ml.fpm import FPGrowth
from pyspark.sql.functions import  col, size

async def main():
    sbx = SbxCore()
    sbx.initialize(os.environ['DOMAIN'], os.environ['APP-KEY'], os.environ['SERVER_URL'])
    login = await sbx.login(os.environ['LOGIN'], os.environ['PASSWORD'], os.environ['DOMAIN'] )
    data = await  sbx.with_model('cart_box_item').fetch_models(['inventory.masterlist', 'cart_box'])\
    .set_page_size(1000)\
    .and_where_is_not_null('cart_box.purchase').find()
    d = {}
    errors = []
    for item in data['results']:
        try:
            item['name'] = data['fetched_results']['add_masterlist'][data['fetched_results']['inventory'][item['inventory']]['masterlist']]['search_name']
            if data['fetched_results']['cart_box'][item['cart_box']]['purchase'] not in d:
                d[data['fetched_results']['cart_box'][item['cart_box']]['purchase']] = {}
            d[data['fetched_results']['cart_box'][item['cart_box']]['purchase']][item['name']] = 1
        except Exception as inst:
            errors.append(inst)
    # print('errors')
    # print(len(errors))
    varieties = [(key, [k for k, val in value.items()]) for key, value in d.items()]
    varieties = list(filter(lambda t: len(t[1]) > 6, varieties))
    print(len(varieties))

    spark = SparkSession.builder.appName("SparkExample").getOrCreate()
    df = spark.createDataFrame(varieties, ["id", "items"])
    fpGrowth = FPGrowth(itemsCol="items", minSupport=0.1, minConfidence=0.6)
    model = fpGrowth.fit(df)
    # Display frequent itemsets.
    model.freqItemsets.sort("freq", ascending=False).show(10, False)

    # Display generated association rules.
    model.associationRules.sort("confidence", ascending=True).show(10)
    model.transform(df).withColumn("item_size", size(col("items")))\
        .withColumn("prediction_size", size(col("prediction")))\
        .orderBy(["item_size","prediction"],ascending=[1,0]).drop('items').show(10, False)
    # $example off$
    spark.stop()


loop = asyncio.new_event_loop()
loop.run_until_complete(main())