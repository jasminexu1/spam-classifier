import org.apache.log4j._
import org.apache.hadoop.fs._
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.rogach.scallop._
import scala.math.exp

class ApplySpamConf(args: Seq[String]) extends ScallopConf(args) {
  mainOptions = Seq(input, model, output)
  val input = opt[String](descr = "input path", required = true)
  val output = opt[String](descr = "output path", required = true)
  val model = opt[String](descr = "model path", required = true)
  verify()
}

object ApplySpamClassifier {
  val log = Logger.getLogger(getClass.getName())

  def main(argv: Array[String]) {
    val args = new ApplySpamConf(argv)

    log.info("Input: " + args.input())
    log.info("Model: " + args.model())
    log.info("Output: " + args.output())

    val spark = SparkSession.builder().appName("ApplySpamClassifier").getOrCreate()
    val sc = spark.sparkContext

    val outputDir = new Path(args.output())
    FileSystem.get(sc.hadoopConfiguration).delete(outputDir, true)

    val modelData = sc.textFile(args.model() + "/part-00000").map(line => {
      val parts = line.stripPrefix("(").stripSuffix(")").split(",")
      (parts(0).toInt, parts(1).toDouble)
    }).collectAsMap()
    
    val w = sc.broadcast(modelData)

    def spamminess(features: Array[Int]): Double = {
      features.map(f => w.value.getOrElse(f, 0.0)).sum
    }

    val result = sc.textFile(args.input()).map(line => {
      val parts = line.split(" ")
      val docid = parts(0)
      val actualLabel = parts(1)
      val features = parts.drop(2).map(_.toInt)
      val score = spamminess(features)
      val predictedLabel = if (score > 0) "spam" else "ham"
      (docid, actualLabel, score, predictedLabel)
    })

    result.saveAsTextFile(args.output())
  }
}

