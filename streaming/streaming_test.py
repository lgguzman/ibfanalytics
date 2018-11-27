import sys, os
from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
if __name__ == "__main__":
    conf = SparkConf().set("spark.jars", "/Users/lgguzman/Documents/PythtonProjects/datamining/ibf-analytics/src/spark-streaming-kafka-0-8.jar")
    sc = SparkContext(appName="prueba", conf=conf)
    ssc = StreamingContext(sc, 2)
    brokers = "localhost:9092"
    topic = "new_topic"
    kvs = KafkaUtils.createDirectStream(ssc, [topic],{"metadata.broker.list": brokers})
    lines = kvs.map(lambda x: x[1])
    counts = lines.flatMap(lambda line: line.split(" ")) \
                  .map(lambda word: (word, 1)) \
                  .reduceByKey(lambda a, b: a+b)
    counts.pprint()
    ssc.start()
    ssc.awaitTermination()