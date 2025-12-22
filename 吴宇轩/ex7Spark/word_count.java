package bigData.spark;

import org.apache.spark.sql.SparkSession;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaPairRDD;
import scala.Tuple2;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.regex.Pattern;

public class word_count {
	// 分词正则（按非字母数字分割）
    private static final Pattern WORD_PATTERN = Pattern.compile("[^a-zA-Z0-9]+");
    // 英文停用词集合
    private static final Set<String> STOP_WORDS = new HashSet<>(Arrays.asList(
            "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
            "in", "on", "at", "to", "for", "of", "with", "by", "I", "you", "he", "she"
    ));

    public static void main(String[] args) {
        // 1. 初始化SparkSession
        SparkSession spark = SparkSession.builder()
                .appName("JavaSparkWordCountAdvanced")
                .master("local[*]")
                .getOrCreate();
        // 2. 读取文本文件
        String inputPath = "/home/hadoop/Desktop/input.txt";
        JavaRDD<String> lines = spark.read().textFile(inputPath).javaRDD();
        // 3. 数据预处理（清洗+分词+过滤停用词）
        JavaRDD<String> words = lines
                // 去除首尾空格
                .map(String::trim)
                // 过滤空行
                .filter(line -> !line.isEmpty())
                // 转小写
                .map(String::toLowerCase)
                // 分词
                .flatMap(line -> Arrays.asList(WORD_PATTERN.split(line)).iterator())
                // 过滤空字符串 + 过滤停用词 + 过滤单字符无意义词
                .filter(word -> !word.isEmpty() 
                        && !STOP_WORDS.contains(word) 
                        && word.length() > 1);
        // 4. 词频统计 + 降序排序
        JavaPairRDD<String, Integer> sortedWordCounts = words
                .mapToPair(word -> new Tuple2<>(word, 1))
                .reduceByKey(Integer::sum)
                .mapToPair(Tuple2::swap)
                .sortByKey(false)
                .mapToPair(Tuple2::swap);
        // 5. 输出结果
        System.out.println("过滤停用词后的词频统计（前10）：");
        sortedWordCounts.take(10).forEach(tuple -> 
            System.out.println(tuple._1() + ": " + tuple._2())
        );
        spark.stop();
    }
}
