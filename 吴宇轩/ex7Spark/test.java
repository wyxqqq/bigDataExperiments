package bigData.spark;

import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
// 静态导入 Spark SQL 函数（col/sum），简化代码
import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.sum;

public class test {
    public static void main(String[] args) {
        // 1. 初始化 SparkSession
        SparkSession spark = SparkSession.builder()
                .appName("SimpleSparkApp")
                .master("local[*]")
                .getOrCreate();

        // 2. 加载 CSV 数据（开启 header + 自动推断 Schema）
        Dataset<Row> data = spark.read()
                .option("header", "true")     
                .option("inferSchema", "true") 
                .csv("/home/hadoop/Desktop/sales_data.csv"); // 数据文件路径

        // 3. 数据处理
        Dataset<Row> result = data
                // 过滤
                .filter(col("product_category").equalTo("Clothing"))
                // 按 date 分组
                .groupBy("date")
                // 求和 revenue
                .agg(sum("revenue").alias("sum_revenue"));

        // 4. 显示结果
        result.show();

        // 5. 关闭 SparkSession
        spark.stop();
    }
}