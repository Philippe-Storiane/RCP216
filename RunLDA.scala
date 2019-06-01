package com.rcp216.racineTopic

import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession

import org.apache.spark.ml.clustering.LDA
import org.apache.spark.ml.clustering.DistributedLDAModel
import org.apache.spark.ml.clustering.KMeansParams

object RunLDA {
  
  def searchClusterSize( minCluster: Int, maxCluster:Int, sc:org.apache.spark.SparkContext, spark: org.apache.spark.sql.SparkSession) = {
    var contentExtractor = new ContentExtractor()
    var paragraphs = contentExtractor.extractContent(sc)
    val (docDF, vocabulary) = contentExtractor.extractDataFrame(paragraphs, sc, spark)
    val ldaTrain = docDF.sample( false, 0.8)
    val ldaTest = docDF.except( ldaTrain )
    val bWordEmbeddings = RunWord2Vec.loadWordEmbeddings(sc)
    val logit = for ( nbClusters <- minCluster to maxCluster ) yield {
      println("Computing LDA for " + nbClusters + " clusters")
      val ( ldaParagraphs, ldaModel) = computeLDA(  ldaTrain, nbClusters )
      val ( kmeansParagraphs, kmeansModel ) = computeKMeans( ldaParagraphs, nbClusters)
      val ldaPerplexity = ldaModel.logPerplexity( ldaTest)
      val ldaLikehood = ldaModel.logLikelihood( ldaTest)
      val kmeansWSSE = kmeansModel.computeCost( kmeansParagraphs)
      val topics = ldaModel.describeTopics(10)
      val topicsDump = topics
        .select("topic","termIndices")
        .rdd
        .map( row => ( row.getAs[Int](0), row.getAs[scala.collection.mutable.WrappedArray[ Int]](1).toSeq.toArray))
        .collect()
      val topicsWord2vec = CoherenceMeasure.word2vec(topicsDump, vocabulary, bWordEmbeddings)
      val word2vec = topicsWord2vec.map( row => row._2).sum / nbClusters
      ( nbClusters, ldaPerplexity, ldaLikehood, kmeansWSSE, word2vec)
    }
    saveLogit("lda-log.csv", logit.toArray)
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
      var topics = ldaModel.describeTopics(10)
      val topicIndices:org.apache.spark.sql.DataFrame = ldaModel.describeTopics(maxTermsPerTopic = 10)

      val wordsWithWeights = org.apache.spark.sql.functions.udf( (x : scala.collection.mutable.WrappedArray[Int],
                             y : scala.collection.mutable.WrappedArray[Double]) => 
        { x.map(i => vocab(i)).zip(y)}
      )
      
      val topics2 = ldaModel.describeTopics(10)
        .withColumn("topicWords", 
          wordsWithWeights(org.apache.spark.sql.functions.col("termIndices"), org.apache.spark.sql.functions.col("termWeights"))
        )
      topics2.select("topicWords").show(false)
      /*
      ldaModel.describeTopics(maxTermsPerTopic = 10).foreach( row => {        
        val topic = row.getAs[Int](0)
        val topicIndices = row.getAs[scala.collection.mutable.WrappedArray[Int]](1)
        val topicWeights = row.getAs[scala.collection.mutable.WrappedArray[Double]](2)
        println("Topic " + topic )
        topicIndices.zipWithIndex.foreach{
          case ( topicId, topicIndex) => {
            print( "( " + vocab(topicId) + " " + topicWeights( topicIndex) + ")")
          }          
        }
       })
       */
  }
   
  def run( k:Int, top: Int, sc:org.apache.spark.SparkContext, spark: org.apache.spark.sql.SparkSession ) {
    var contentExtractor = new ContentExtractor()
    var paragraphs = contentExtractor.extractContent(sc)
    val (docDF, vocabulary) = contentExtractor.extractDataFrame(paragraphs, sc, spark)
    val lda = new org.apache.spark.ml.clustering.LDA()
      .setK( 10 )
      .setMaxIter(500)
      .setFeaturesCol("tf")
    val ldaModel = lda.fit( docDF )
    val ldaParagraphs = ldaModel.transform( docDF )
    
    val logLikehood = ldaModel.logLikelihood( docDF )
    val logPerplexity = ldaModel.logPerplexity( docDF )
    
    val topics = ldaModel.describeTopics( 3 )
    topics.show( false )
    
    ldaModel.save( "ldaModel")
    val ldaDistributedModel = org.apache.spark.ml.clustering.DistributedLDAModel.load("ldaModel")
    
  }
  
  def saveLogit( path: String, wsse : Array[ (Int, Double, Double, Double, Double) ]) = {
    val ps = new java.io.PrintStream(new java.io.FileOutputStream(path))
    ps.println("topic\tlogLikehood\tlogPerplexity\twsse\tword2vec")
    wsse.foreach( cost => ps.println(cost._1 + "\t"+ + cost._2  +"\t" + cost._3 + "\t" + cost._4 + "\t" + cost._5))
    ps.close()
  }
}