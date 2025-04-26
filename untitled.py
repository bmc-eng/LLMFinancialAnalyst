from pyspark.sql import SparkSession, Window
import pyspark.sql.functions as F
from pyspark.sql.types import TimestampType

from utils.s3_helper import S3Helper
from IPython.display import display

import os
import boto3

class FinancialNewsRequester:


    def __init__(self, index_members, start_date):
        self.spark = self._get_spark_session(executors="100",
                                             executor_memory="8g",
                                             executor_cores="2")
        

    
    def build_news_dataset(self, index_members, start_date):
        df = self._load_all_headlines()
        df1 = self._filter_news_headlines(df, index_members, start_date)
        df2 = self._drop_duplicates(df1)
        pdf = self._sort_headlines(df2)
        self.spark.stop()
        return pdf


    def _drop_duplicates(self, df1):
        window = Window.partitionBy("SUID").orderBy(F.col("TimeOfArrival").asc())
        
        df2 = (
            df1.withColumn("row", F.row_number().over(window))
            .filter(F.col("row") == 1)
            .drop("row")
        )
        
        df2 = df2.withColumn("day", F.to_date(F.col("TimeOfArrival")))
        window = Window.partitionBy("day", "Headline").orderBy(F.col("TimeOfArrival").asc())
        df2 = (
            df2.withColumn("row", F.row_number().over(window))
            .filter(F.col("row") == 1)
            .drop("row", "day")
        )
        return df2.cache()

        

    def _filter_news_headlines(self, df, index_members, start_date):
        # Filter for just BBG news or include all news articles in the analysis.
        wire_filter = (F.col("WireName") == "BN") | (F.col("WireName") == "BFW")
        
        filters = (
            # topic_filter
            wire_filter
            & (F.col("LanguageString") == "ENGLISH")
            & (F.length(F.col("Headline")) > 25)
            & (F.col("TimeOfArrival") >= start_date)
            & (F.col("Assigned_ID_BB_GLOBAL").isin(index_members))
            #& (F.col("Headline").startswith("*"))
        )
        
        df = df.withColumn("TimeOfArrival", F.col("TimeOfArrival").cast(TimestampType()))
        df1 = df.filter(filters)
        
        return df1
    
    
    def _load_all_headlines(self, spark):
        bucket_name = "bquant-data-textual-analytics-tier-1"
        bucket = boto3.resource("s3").Bucket(bucket_name)
        files = [file.key for file in bucket.objects.all()]
        
        files_csv = [
            f"s3://{bucket_name}/{file}"
            for file in files
            if "EID80001" in file and "csv" in file
        ]
        
        df = (
            spark.read.option("header", "true")
            .option("multiLine", "true")
            .option("escape", "")
            .csv(files_csv)
        )
        return df
    
    
    def _sort_headlines(self, df2):
        pdf = (
            df2.select(
                "SUID", "Headline", "TimeOfArrival", "Assigned_ID_BB_GLOBAL"
            )
            .toPandas()
            .sort_values(by="TimeOfArrival")
            .reset_index(drop=True)
            .copy()
    
        )
        pdf["Headline"] = pdf["Headline"].str.lower()
    
        return pdf
    
    
    def _get_spark_session(
        executors="10",
        executor_memory="8g",
        driver_memory="32g",
        executor_cores="2",
        driver_max_result_size="1024M",
        executor_memory_overhead="2g",
        task_cpus="1",
    ):
    
        spark = (
            SparkSession.builder.config("spark.driver.memory", driver_memory)
            .config("spark.driver.maxResultSize", driver_max_result_size)
            .config("spark.executor.memoryOverhead", executor_memory_overhead)
            .config("spark.executor.instances", executors)
            .config("spark.executor.memory", executor_memory)
            .config("spark.executor.cores", executor_cores)
            .config("spark.task.cpus", task_cpus)
            .config("spark.sql.execution.arrow.enabled", "true")
            .config("spark.shuffle.file.buffer", "1m")
            .config("spark.file.transferTo", "False")
            .config("spark.shuffle.unsafe.file.output.buffer", "1m")
            .config("spark.io.compression.lz4.blockSize", "512k")
            .config("spark.shuffle.service.index.cache.size", "1g")
            .config("spark.shuffle.registration.timeout", "120000ms")
            .config("spark.shuffle.registration.maxAttempts", "3")
            .config("spark.sql.windowExec.buffer.spill.threshold", "1000000")
            .config("spark.sql.windowExec.buffer.in.memory.threshold", "1000000")
            .getOrCreate()
        )
    
        display(spark)
    
        return spark