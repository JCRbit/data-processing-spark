// Databricks notebook source
import org.apache.spark.sql.SparkSession

// Create SparkSession object
val spark = SparkSession.builder()
      .appName("World Happineess 2005-2021")
      .config("spark.sql.warehouse.dir", "<path>/spark-warehouse")
      .getOrCreate()

// COMMAND ----------

import org.apache.spark.sql.types.{StructType, StringType, IntegerType, DoubleType};

// Define data schemas
val schemaWorldHappiness2005_2020 = new StructType()
      .add("Country name", StringType, true)
      .add("Year", IntegerType, true)
      .add("Life Ladder", DoubleType, true)
      .add("Log GDP per capita", DoubleType, true)
      .add("Social support", DoubleType, true)
      .add("Healthy life expectancy at birth", DoubleType, true)
      .add("Freedom to make life choices", DoubleType, true)
      .add("Generosity", DoubleType, true)
      .add("Perceptions of corruption", DoubleType, true)
      .add("Positive affect", DoubleType, true)
      .add("Negative affect", DoubleType, true)

val schemaWorldHappiness2021 = new StructType()
      .add("Country name", StringType, true)
      .add("Regional indicator", StringType, true)
      .add("Ladder score", DoubleType, true)
      .add("Standard error of ladder score", DoubleType, true)
      .add("upperwhisker", DoubleType, true)
      .add("lowerwhisker", DoubleType, true)
      .add("Logged GDP per capita", DoubleType, true)
      .add("Social support", DoubleType, true)
      .add("Healthy life expectancy", DoubleType, true)
      .add("Freedom to make life choices", DoubleType, true)
      .add("Generosity", DoubleType, true)
      .add("Ladder score in Dystopia", DoubleType, true)
      .add("Explained by: Log GDP per capita", DoubleType, true)
      .add("Explained by: Social support", DoubleType, true)
      .add("Explained by: Healthy life expectancy", DoubleType, true)
      .add("Explained by: Freedom to make life choices", DoubleType, true)
      .add("Explained by: Generosity", DoubleType, true)
      .add("Explained by: Perceptions of corruption", DoubleType, true)
      .add("Dystopia + residual", DoubleType, true)

// COMMAND ----------

// Read csv data files and create DataFrames
val dfWorldHappiness2005_2020 = spark.read.options(Map("delimiter"->",", "header"->"true"))
      .schema(schemaWorldHappiness2005_2020)
      .csv("dbfs:/FileStore/world_happiness_report.csv")

val dfWorldHappiness2021 = spark.read.options(Map("delimiter"->",", "header"->"true"))
      .schema(schemaWorldHappiness2021)
      .csv("dbfs:/FileStore/world_happiness_report_2021.csv")

// COMMAND ----------

import org.apache.spark.sql.functions.col

// Create Country-Region DataFrame
val dfCountryRegion = dfWorldHappiness2021.groupBy("Country name", "Regional indicator").count()
      .withColumnRenamed("Country name", "Country") 

val colsWorldHappiness2005_2020: List[String] = List(
      "Country name",
      "Regional indicator",
      "Year",
      "Life Ladder",
      "Log GDP per capita",
      "Healthy life expectancy at birth")

// Create Region column, select and rename target columns
val dfWorld2005_2020 = dfWorldHappiness2005_2020.join(dfCountryRegion, dfWorldHappiness2005_2020("Country name") === dfCountryRegion("Country"), "left") 
      .select(colsWorldHappiness2005_2020.map(c => col(c)):_*) 
      .withColumnRenamed("Country name", "Country") 
      .withColumnRenamed("Regional indicator", "Region") 
      .withColumnRenamed("Life Ladder", "Ladder Score") 
      .withColumnRenamed("Healthy life expectancy at birth", "Healthy life expectancy")

dfWorld2005_2020.printSchema()

// COMMAND ----------

import org.apache.spark.sql.functions.lit

val colsWorldHappiness2021: List[String] = List(
      "Country name",
      "Regional indicator",
      "Year",
      "Ladder score",
      "Logged GDP per capita",
      "Healthy life expectancy")

// Create Year column and select taget columns
val dfWorld_2021 = dfWorldHappiness2021.withColumn("Year", lit(2021)) 
      .select(colsWorldHappiness2021.map(c => col(c)):_*) 

dfWorld_2021.printSchema()

// COMMAND ----------

// Union 2005-2020 and 2021 DataFrames
val dfWorldHappiness2005_2021 = dfWorld2005_2020.union(dfWorld_2021)
      .sort("Country", "Year")
    
dfWorldHappiness2005_2021.show(15)

// COMMAND ----------

import org.apache.spark.sql.functions.sum

// Missing values
dfWorldHappiness2005_2021.select(dfWorldHappiness2005_2021.columns.map(c => sum(col(c).isNull.cast(IntegerType)).alias(c)): _*).show()

// COMMAND ----------

// Countries with no Region value
dfWorldHappiness2005_2021.filter(col("Region").isNull)
    .groupBy("Country").count()
    .drop("count")
    .show()

// COMMAND ----------

// DBTITLE 1,1. ¿Cuál es el país más “feliz” del 2021 según la data?
import org.apache.spark.sql.functions.desc

val dfHappiness2021 = dfWorldHappiness2005_2021.filter(col("Year") === 2021) 
dfHappiness2021.select("Country", "Region", "Year", "Ladder Score") 
      .sort(desc("Ladder Score")) 
      .show(10, false)  // Finland

// COMMAND ----------

// DBTITLE 1,2. ¿Cuál es el país más “feliz” del 2021 por continente según la data?
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.row_number

val windowRegion = Window.partitionBy("Region").orderBy(desc("Ladder Score"))
dfHappiness2021.withColumn("row", row_number.over(windowRegion)) 
      .filter(col("row") === 1).drop("row") 
      .sort(desc("Ladder Score")) 
      .select("Country", "Region", "Ladder Score") 
      .show(false)

// COMMAND ----------

// DBTITLE 1,3. ¿Cuál es el país que más veces ocupó el primer lugar en todos los años?
val windowYear = Window.partitionBy("Year").orderBy(desc("Ladder Score"))
dfWorldHappiness2005_2021.withColumn("row", row_number.over(windowYear))
      .filter(col("row") === 1).drop("row")
      .groupBy("Country").count()
      .sort(desc("count"))
      .show(false)  // Finland & Denmark

// COMMAND ----------

// DBTITLE 1,4. ¿Qué puesto de Felicidad tiene el país con mayor GDP del 2020?
val window = Window.orderBy(desc("Ladder Score"))
dfWorldHappiness2005_2021.filter(col("Year") === 2020)
    .withColumn("Hapinness Rank", row_number.over(window))
    .drop("Healthy life expectancy")
    .sort(desc("Log GDP per capita"))
    .show(10)  // 13

// COMMAND ----------

// DBTITLE 1,5. ¿En que porcentaje ha variado a nivel mundial el GDP promedio del 2020 respecto al 2021? ¿Aumentó o disminuyó?
import org.apache.spark.sql.functions.{exp, avg}

val dfHappiness2020 = dfWorldHappiness2005_2021.filter(col("Year") === 2020)
val gdp2020: Double = dfHappiness2020.select(avg(exp("Log GDP per capita")).alias("GDPpc")).first().getAs[Double]("GDPpc")
val gdp2021: Double = dfHappiness2021.select(avg(exp("Log GDP per capita")).alias("GDPpc")).first().getAs[Double]("GDPpc")
val relVariation: Double = gdp2021 / gdp2020 - 1
println(f"${relVariation * 100}%.2f%%")  // -13.01%

// COMMAND ----------

// DBTITLE 1,6. ¿Cuál es el país con mayor expectativa de vida? Y ¿Cuánto tenía en ese indicador en el 2019?
val dfLifeExpectancy2021 = dfHappiness2021.sort(desc("Healthy life expectancy"))
dfLifeExpectancy2021.show(10, false)

val highestLifeExpectancyCountry = dfLifeExpectancy2021.first()(0)  // Singapore
dfWorldHappiness2005_2021.filter((col("Country") === highestLifeExpectancyCountry) && (col("Year") === 2019)) 
      .select("Country", "Year", "Healthy life expectancy") 
      .show()  // 77.1 years
