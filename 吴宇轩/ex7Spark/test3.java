package bigData.spark;

import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.countDistinct;
import static org.apache.spark.sql.functions.desc;
import static org.apache.spark.sql.functions.round;
import static org.apache.spark.sql.functions.sum;
import static org.apache.spark.sql.functions.to_date;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class test3 {
	public static void main(String[] args) {
        // 1. 初始化SparkSession
        SparkSession spark = SparkSession.builder()
                .appName("DailyAvgRevenueBySubCategory")
                .master("local[*]")  
                .getOrCreate();

        // 2. 读取数据并解析日期
        Dataset<Row> df = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv("/home/hadoop/Desktop/sales_data.csv")
                .withColumn("date_parsed", to_date(col("Date"), "yyyy/M/d"));

        // 3. 按产品子类分组，统计总销售额、总营业天数
        Dataset<Row> subcategoryStatsDF = df.groupBy("Sub_Category")
                .agg(
                    sum("Revenue").alias("total_revenue"),
                    countDistinct("date_parsed").alias("total_days")
                );

        // 4. 计算日均销售额（保留2位小数）
        Dataset<Row> dailyAvgRevenueDF = subcategoryStatsDF.withColumn(
            "daily_avg_revenue",
            round(col("total_revenue").divide(col("total_days")), 2)
        );

        // 5. 展示结果（降序）
        dailyAvgRevenueDF.orderBy(desc("daily_avg_revenue")).show();

        spark.stop();
    }
}
