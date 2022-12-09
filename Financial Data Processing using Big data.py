from os.path import abspath
from pyspark.sql import SparkSession
import pyspark
from functools import reduce
from pyspark.sql.types import StringType, DateType, FloatType
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark import SparkContext, SparkConf

# warehouse_location points to the default location for managed databases and tables
warehouse_location = abspath('spark-warehouse')

# Create spark session with hive enabled
spark = SparkSession \
    .builder \
    .appName("SparkByExamples.com") \
    .config("spark.sql.warehouse.dir", warehouse_location) \
    .config("spark.sql.catalogImplementation", "hive") \
    .enableHiveSupport() \
    .getOrCreate()

#read data from hdfs

good = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("hdfs:///user/maria_dev/spark_files/good_companies.csv")

distress = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("hdfs:///user/maria_dev/spark_files/listed_mfg_newest.csv")

#data processing
good_columns = list(map(lambda item: item.replace(".", "").upper(), good.columns))
distress_columns = list(map(lambda item: item.replace(".", "").upper(), distress.columns))

good_df = reduce(lambda data, idx: data.withColumnRenamed(good.columns[idx], good_columns[idx]),
                 range(len(good_columns)), good)
distress_df = reduce(lambda data, idx: data.withColumnRenamed(distress.columns[idx], distress_columns[idx]),
                     range(len(distress_columns)), distress)
#common columns in both data
columns = list(set(good_df.columns).intersection(set(distress_df.columns)))
goods_final = good_df.select(*columns)
distress_final = distress_df.select(*columns)

df = goods_final.union(distress_final)

#changing data type of each column in required format
df = df \
    .withColumn("BS_Total Reserves",
                df["BS_Total Reserves"]
                .cast(FloatType())) \
    .withColumn("BS_Securities Premium",
                df["BS_Securities Premium"]
                .cast(FloatType())) \
    .withColumn("BS_Govt Grant",
                df["BS_Govt Grant"]
                .cast(FloatType())) \
    .withColumn("BS_Revaluation reserve",
                df["BS_Revaluation reserve"]
                .cast(FloatType())) \
    .withColumn("BS_   Revaluation reserve - Assets",
                df["BS_   Revaluation reserve - Assets"]
                .cast(FloatType())) \
    .withColumn("Nmonth_Close",
                df["Nmonth_Close"]
                .cast(FloatType())) \
    .withColumn("BS_Number of Equity Shares Paid Up",
                F.regexp_replace("BS_Number of Equity Shares Paid Up", ",", "").cast(FloatType())) \
    .withColumn("FH_Total Assets",
                F.regexp_replace("FH_Total Assets", ",", "").cast(FloatType()))

#calculating required field for ratios
df = df.na.fill(value=0.0,
                subset=["BS_Total Reserves", 'BS_Securities Premium', 'BS_Govt Grant', 'BS_Revaluation reserve',
                        'BS_   Revaluation reserve - Assets', 'BS_Subsidy Reserve (Central / State)'])
df = df.withColumn('retain_earnings', (
            df['BS_Total Reserves'] - df['BS_Securities Premium'] - df['BS_Govt Grant'] - df['BS_Revaluation reserve'] -
            df['BS_   Revaluation reserve - Assets'] - df['BS_Subsidy Reserve (Central / State)']))
df = df.withColumn('MCAP', ((df['Nmonth_Close'] * df['BS_Number of Equity Shares Paid Up']) / 10000000))
df = df.withColumn('BS_Total Liabilities', (df['FH_Total Assets'] - df['FH_Total Current Liabilities']))
df = df.withColumn('net_Working_Capital', (df['BS_Total Current Assets'] - df['FH_Total Current Liabilities']))


#calculating moving average of data
windowSpec = Window().partitionBy(['Company Name', 'FIN']).orderBy('FH_Year End').rowsBetween(-3, 0)
df = df.withColumn("MCAP_moving", F.mean("MCAP").over(windowSpec))
df = df.withColumn("Total_Applications_Nf_moving", F.mean("FH_Total Assets").over(windowSpec))
df = df.withColumn("Net_sales_moving", F.mean("PL_Net Sales").over(windowSpec))
df = df.withColumn("BS_Total Liabilities_moving", F.mean("BS_Total Liabilities").over(windowSpec))
df = df.withColumn("FH_PBIT_moving", F.mean("FH_PBIT").over(windowSpec))
df = df.withColumn("FH_Net Worth_moving", F.mean("FH_Net Worth").over(windowSpec))
df = df.withColumn("retain_earnings_moving", F.mean("retain_earnings").over(windowSpec))
df = df.withColumn("net_Working_Capital_moving", F.mean("net_Working_Capital").over(windowSpec))

#ratios calculation

final = df.withColumn('A', (df['net_Working_Capital_moving'] / df['Total_Applications_Nf_moving'])) \
    .withColumn('B', (df['retain_earnings_moving'] / df['Total_Applications_Nf_moving'])) \
    .withColumn('C', (df['FH_PBIT_moving'] / df['Total_Applications_Nf_moving'])) \
    .withColumn('D', (df['MCAP_moving'] / df['Total_Applications_Nf_moving'])) \
    .withColumn('E', (df['Net_sales_moving'] / df['Total_Applications_Nf_moving']))

final = final.select(['COMPANY NAME', 'FIN', 'RATING', 'FH_Year End', 'A', 'B', 'C', 'D', 'E'])
final=final.withColumnRenamed("COMPANY NAME","company_name").withColumnRenamed("FH_Year End","FH_Year_End")

# Create Hive Internal table
final.write.mode('overwrite') \
          .partitionBy('FIN') \
          .saveAsTable("altman2")

df = spark.read.table("altman2")
df.show()