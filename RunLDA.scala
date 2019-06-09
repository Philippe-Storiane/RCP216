package com.rcp216.racineTopic

import org.apache.spark.SparkContext
import org.apache.spark.sql.functions._
import org.apache.spark.sql.SparkSession

import org.apache.spark.ml.clustering.LDA
import org.apache.spark.ml.clustering.DistributedLDAModel
import org.apache.spark.ml.clustering.KMeansParams

class RunLDA extends AbstractRun {
  
  def searchClusterSize( minCluster: Int, maxCluster:Int, sc:org.apache.spark.SparkContext, spark: org.apache.spark.sql.SparkSession) = {
    var contentExtractor = new ContentExtractor()
    var paragraphs = contentExtractor.extractContent(sc)
    val (docDF, vocabulary) = contentExtractor.extractDataFrame(paragraphs, sc, spark)
    val ldaTrain = docDF.sample( false, 0.8)
    val ldaTest = docDF.except( ldaTrain )
    val bWordEmbeddings = CoherenceMeasure.loadWordEmbeddings(sc)
    val corpusPMI = CoherenceMeasure.preprocessUMass( docDF, vocabulary.length )
    val logit = for ( nbClusters <- minCluster to maxCluster ) yield {
    println("Computing LDA for " + nbClusters + " clusters")
    val ( ldaParagraphs, ldaModel) = computeLDA(  ldaTrain, nbClusters )
    val ( kmeansParagraphs, kmeansModel ) = computeKMeans( ldaParagraphs, nbClusters)
    val ldaPerplexity = ldaModel.logPerplexity( ldaTest)
    val ldaLikehood = ldaModel.logLikelihood( ldaTest)
    val kmeansWSSE = kmeansModel.computeCost( kmeansParagraphs)
    val topicsDump = extractTopics( ldaModel, 10)
    val topicsWord2vec = CoherenceMeasure.word2vec(topicsDump, vocabulary, bWordEmbeddings)
    val word2vec = topicsWord2vec.map( _._2).sum / nbClusters
    val uMass = CoherenceMeasure.uMass(topicsDump, corpusPMI).map(_._2).sum / nbClusters
      ( nbClusters, ldaPerplexity, ldaLikehood, kmeansWSSE, word2vec, uMass)
      
    }
    saveLogit("lda-log.csv", logit.toArray)
  }
  
  def extractTopics( ldaModel: org.apache.spark.ml.clustering.LDAModel, top:Int) = {
      val topics = ldaModel.describeTopics(top)
      val topicsDump = topics
        .select("topic","termIndices")
        .rdd
        .map( row => ( row.getAs[Int](0), row.getAs[scala.collection.mutable.WrappedArray[ Int]](1).toSeq.toArray))
        .collect()
      topicsDump
  }
  
  
  def computeKMeans( docDF: org.apache.spark.sql.DataFrame, nbClusters: Int) = {
    val nbIterations = 100
       val kmeans = new org.apache.spark.ml.clustering.KMeans()
      .setK( nbClusters )
      .setMaxIter( nbIterations)
      .setFeaturesCol("topicDistribution")
      .setInitMode("k-means||")
      .setPredictionCol("topic")
      
      val kmeansModel = kmeans.fit( docDF )
      val kmeanParagraphs = kmeansModel.transform( docDF )
      ( kmeanParagraphs, kmeansModel)
  }
  
  def computeLDA( docDF: org.apache.spark.sql.DataFrame, nbClusters: Int) = {
    val lda = new org.apache.spark.ml.clustering.LDA()
      .setK( nbClusters )
      .setMaxIter(500)
      .setFeaturesCol("tf")
    val ldaModel = lda.fit( docDF )
    val ldaParagraphs = ldaModel.transform( docDF )
    ( ldaParagraphs, ldaModel )
  }
  
   def analyzeCluster( nbCluster:Int, sc:org.apache.spark.SparkContext, spark: org.apache.spark.sql.SparkSession) = {
      val contentExtractor = new ContentExtractor()
      var paragraphs = contentExtractor.extractContent(sc)
      var ( paragraphsDF, vocab )  = contentExtractor.extractDataFrame( paragraphs, sc, spark)
    
      val ( ldaParagraphs, ldaModel ) = computeLDA(  paragraphsDF, nbCluster )
 
      val wordsWithWeights = org.apache.spark.sql.functions.udf( (x : scala.collection.mutable.WrappedArray[Int],
                             y : scala.collection.mutable.WrappedArray[Double]) => 
        { x.map(i => vocab(i)).zip(y)}
      )
      
      val topics2 = ldaModel.describeTopics(10)
        .withColumn("topicWords", 
          wordsWithWeights(org.apache.spark.sql.functions.col("termIndices"), org.apache.spark.sql.functions.col("termWeights"))
        )
      topics2.select("topicWords").show(false)

      val topicDistribution = ldaParagraphs.select("topicDistribution").withColumn("id",org.apache.spark.sql.functions.monotonicallyIncreasingId)
      val topicMap = topicDistribution.collect().map( row => {
          val topic = row.getAs[org.apache.spark.ml.linalg.DenseVector]("topicDistribution").toArray
          val id = row.getAs[Long]("id")
          val topicDesc = topic.zipWithIndex.sortWith( _._1 > _._1).take(5).map(_.swap)
          ( id, topicDesc )
        }
      )
      saveTopicMap("lda-topicMap.csv", topicMap)
      
      val topicWordsDF = ldaModel.describeTopics( 10 )
      val topicWords = topicWordsDF.collect().map( row => {
          val topic = row.getAs[Int]("topic")
          val termIndices = row.getAs[scala.collection.mutable.WrappedArray[Int]]("termIndices")
          val termWeights = row.getAs[scala.collection.mutable.WrappedArray[Double]]("termWeights")
          val wordDesc = termIndices.map( word => vocab( word)).zip(termWeights)
          ( topic, wordDesc)
        }
      )
      
      saveTopicWords("lda-topicWords.csv", topicWords)
  }
   
 
  
  def saveLogit( path: String, wsse : Array[ (Int, Double, Double, Double, Double, Double) ]) = {
    val ps = new java.io.PrintStream(new java.io.FileOutputStream(path))
    ps.println("topic\tlogLikehood\tlogPerplexity\twsse\tword2vect\tUMass")
    wsse.foreach( row => ps.println( row._1 + "\t" + row._2 + "\t" + row._3 + "\t" + row._4 + "\t" + row._5 + "\t" + row._6))
    ps.close()
  }
  

}