import org.apache.log4j._
import org.apache.hadoop.fs._
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.rogach.scallop._

class ApplyEnsembleSpamConf(args: Seq[String]) extends ScallopConf(args) {
  mainOptions = Seq(input, output, model, method)
  val input = opt[String](descr = "input path", required = true)
  val output = opt[String](descr = "output path", required = true)
  val model = opt[String](descr = "model path", required = true)
  val method = opt[String](descr = "ensemble method (average or vote)", required = true)
  verify()
}

object ApplyEnsembleSpamClassifier {
  val log = Logger.getLogger(getClass.getName())

  def main(argv: Array[String]) {
    val args = new ApplyEnsembleSpamConf(argv)
    log.info("Input: " + args.input())
    log.info("Output: " + args.output())
    log.info("Model: " + args.model())
    log.info("Method: " + args.method())

    val modelPath = args.model() + "/part-*"
    val inputPath = args.input()
    val outputPath = args.output()
    val method = args.method()

    val spark = SparkSession.builder().appName("ApplyEnsembleSpamClassifier").getOrCreate()
    val sc = spark.sparkContext
    val outputDir = new Path(outputPath)
    FileSystem.get(sc.hadoopConfiguration).delete(outputDir, true)

    val models = sc.wholeTextFiles(modelPath)
      .map { case (_, content) =>
        content.split("\n")
          .filter(_.nonEmpty)
          .map(line => {
            val parts = line.stripPrefix("(").stripSuffix(")").split(",")
            (parts(0).trim.toInt, parts(1).trim.toDouble)
          })
          .toMap
      }
      .collect()
    val w = sc.broadcast(models)

    def spamminess(features: Array[Int], model: Map[Int, Double]): Double = {
      var score = 0d
      features.foreach(f => if (model.contains(f)) score += model(f))
      score
    }

    val result = sc.textFile(inputPath).map(line => {
      val parts = line.split(" ")
      val docid = parts(0)
      val actualLabel = parts(1)
      val features = parts.drop(2).map(_.toInt)
      val scores = w.value.map(model => spamminess(features, model))
      val (ensembleScore, prediction) = method match {
        case "average" =>
          val avgScore = scores.sum / scores.length
          (avgScore, if (avgScore > 0) "spam" else "ham")
        case "vote" =>
          val votes = scores.map(score => if (score > 0) 1 else -1)
          val total = votes.sum
          (total.toDouble, if (total > 0) "spam" else "ham")
      }

      (docid, actualLabel, ensembleScore, prediction)
    }).saveAsTextFile(outputPath)
  }
}
