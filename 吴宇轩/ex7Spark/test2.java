package bigData.spark;

import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import static org.apache.spark.sql.functions.*;

public class test2 {
	public static void main(String[] args) {
        // 1. 初始化SparkSession
        SparkSession spark = SparkSession.builder()
                .appName("MaxWeekdayRevenueCategory")
                .master("local[*]")
                .getOrCreate();

        // 2. 读取销售数据，解析日期
        Dataset<Row> df = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv("/home/hadoop/Desktop/sales_data.csv")
                // 统一日期格式
                .withColumn("date_parsed", to_date(col("Date"), "yyyy/M/d"));

        // 3. 筛选工作日数据（周一=2 ~ 周五=6）
        Dataset<Row> weekdayDF = df.filter(dayofweek(col("date_parsed")).between(2, 6));

        // 4. 按产品类别聚合工作日总销售额
        Dataset<Row> categoryRevenueDF = weekdayDF.groupBy("Product_Category")
                .agg(sum("Revenue").alias("total_weekday_revenue"));

        categoryRevenueDF.orderBy(desc("total_weekday_revenue")).show();

        spark.stop();
    }
}
