import org.apache.log4j._
import org.apache.hadoop.fs._
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.rogach.scallop._
import scala.math.exp

class TrainSpamConf(args: Seq[String]) extends ScallopConf(args) {
  mainOptions = Seq(input, model, shuffle)
  val input = opt[String](descr = "input path", required = true)
  val model = opt[String](descr = "model", required = true)
  val shuffle = opt[Boolean](descr = "shuffle", required = false)
  verify()
}

object TrainSpamClassifier{
  val log = Logger.getLogger(getClass().getName())

  def main(argv: Array[String]) {
    val args = new TrainSpamConf(argv)

    log.info("Input: " + args.input())
    log.info("Model: " + args.model())
    log.info("Shuffle: " + args.shuffle())

    val spark = SparkSession.builder().appName("TrainSpamClassifier").getOrCreate()
    val sc = spark.sparkContext
    var textFile = sc.textFile(args.input(), 1)
    val w = scala.collection.mutable.Map[Int, Double]().withDefaultValue(0.0)
    val outputDir = new Path(args.model())
    FileSystem.get(sc.hadoopConfiguration).delete(outputDir, true)

    if (args.shuffle()) {
      textFile = textFile.map(line => (scala.util.Random.nextDouble(), line)).sortByKey().map(_._2)
    }

    def spamminess(features: Array[Int]): Double = {
      var score = 0d
      features.foreach(f => if (w.contains(f)) score += w(f))
      score
    }

    val delta = 0.002
    val trained = textFile.map(line => {
      val parts = line.split(" ")
      val docid = parts(0)
      val isSpam = if (parts(1) == "spam") 1.0 else 0.0
      val features = parts.drop(2).map(_.toInt)
      (0, (docid, isSpam, features))
    }).groupByKey(1)

    trained.flatMap { case (_, instances) =>
      instances.foreach { case (_, isSpam, features) =>
        val score = spamminess(features)
        val prob = 1.0 / (1 + exp(-score))
        features.foreach(f => {
          w(f) += (isSpam - prob) * delta
        })
      }
      w.toList
    }.saveAsTextFile(args.model())
  }
}

