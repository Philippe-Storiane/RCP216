package com.cnam.rcp216.racineTopic

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
    val measures = for ( nbClusters <- minCluster to maxCluster ) yield {
      println("Computing LDA for " + nbClusters + " clusters")
      val ( ldaParagraphs, ldaModel) = computeLDA(  docDF, nbClusters )
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
    saveMeasures("lda-measures-tst.csv", measures.toArray)
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
  
  

  def computeLDA( docDF: org.apache.spark.sql.DataFrame, nbClusters: Int) = {
    var maxIterations = 500
    var docConcentration = -1.0
    var topicConcentration = -1.0
    var optimizer = "online"
    var seed = 1L
    var prop = System.getProperty("rcp216.lda.maxIterations")
    if ( prop != null ) {
      maxIterations = prop.toInt
    }
    prop = System.getProperty("rcp216.lda.topicConcentration")
    if ( prop != null ) {
      topicConcentration = prop.toDouble
    }
    prop = System.getProperty("rcp216.lda.docConcentration")
    if ( prop != null ) {
      docConcentration = prop.toDouble
    }
    prop = System.getProperty("rcp216.lda.seed")
    if ( prop != null ) {
      seed = prop.toLong
    }
    prop = System.getProperty("rcp216.lda.optimizer")
    if ( prop != null ) {
      optimizer = prop
    }
    var lda = new org.apache.spark.ml.clustering.LDA()
      .setK( nbClusters )
//      .setOptimizer("EM")
      .setOptimizer( optimizer )
      .setMaxIter( maxIterations )
      .setSeed( seed )
//      .setDocConcentration( docConcentration )
//      .setTopicConcentration( topicConcentration )
      .setFeaturesCol("tf")
    if ( docConcentration != -1 ) {
      lda = lda.setDocConcentration( docConcentration )
    }
    if ( topicConcentration != - 1 ) {
      lda = lda.setDocConcentration( topicConcentration )
    }
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
      saveTopicMap("lda-topicMap-tst.csv", topicMap)
      
      val topicWordsDF = ldaModel.describeTopics( 10 )
      val topicWords = topicWordsDF.collect().map( row => {
          val topic = row.getAs[Int]("topic")
          val termIndices = row.getAs[scala.collection.mutable.WrappedArray[Int]]("termIndices")
          val termWeights = row.getAs[scala.collection.mutable.WrappedArray[Double]]("termWeights")
          val wordDesc = termIndices.zipWithIndex.map{ case( word, index) => (word, vocab( word), termWeights( index))}
          ( topic, wordDesc)
        }
      )
      
      saveTopicWords("lda-topicWords-tst.csv", topicWords)
  }
   
 def analyzeFeaturedCluster( nbCluster:Int, sc:org.apache.spark.SparkContext, spark: org.apache.spark.sql.SparkSession) = {
     val top = 10 
     val contentExtractor = new ContentExtractor()
     var paragraphs = contentExtractor.extractContent(sc)
     var ( paragraphsDF, vocab )  = contentExtractor.extractDataFrame( paragraphs, sc, spark)
    
     val ( ldaParagraphs, ldaModel ) = computeLDA(  paragraphsDF, 200 )
 
     val ( kmeansParagraphs, kmeanModel ) = computeKMeans( ldaParagraphs, nbCluster)
     val featuredTerms = ldaModel.describeTopics( vocab.size).collect()
     var isCos2Distance = false
     val prop = System.getProperty("rcp216.lda.kmeans.distance")
     if ( "cos2".equals( prop) ) {
       isCos2Distance = true
     }
     val topicWords = kmeanModel.clusterCenters.zipWithIndex.map{
          { case ( cluster, clusterIndex )  => {
              println("cluster " + clusterIndex)
              val clusterArray = cluster.toArray
              val wordsDistance = vocab.zipWithIndex.map {
                case ( word, wordIndex) => {
                  var distance = 0.0
                  clusterArray.zipWithIndex.foreach{
                    case ( clusterFeatureValue, clusterFeatureIndex) => {
                      val featureIndex = featuredTerms( clusterFeatureIndex ).getInt(0)
                      if ( featureIndex != clusterFeatureIndex) {
                        println("WARNING: Feature wrong index")
                      }
                      val wordFeatureIndexes = featuredTerms( clusterFeatureIndex ).getAs[scala.collection.mutable.WrappedArray[Int]](1)
                      val wordFeatureValues = featuredTerms( clusterFeatureIndex ).getAs[scala.collection.mutable.WrappedArray[Double]](2)
                      val wordFeatureIndex = wordFeatureIndexes.indexOf( wordIndex )
                      val wordFeatureValue = wordFeatureValues( wordFeatureIndex)
                      distance += wordFeatureValue * math.log( 2 * wordFeatureValue) -
                        wordFeatureValue * math.log( wordFeatureValue + clusterFeatureValue) +
                        clusterFeatureValue * math.log( 2 * clusterFeatureValue) -
                        clusterFeatureValue * math.log( wordFeatureValue + clusterFeatureValue) 
                    }
                  }
                  (wordIndex, word, distance  )                  
                }
              }
              val topWordDistance = wordsDistance.sortWith( _._3 < _._3).take(top)
              val topWords:scala.collection.mutable.WrappedArray[(Int, String, Double)] = topWordDistance
              ( clusterIndex, topWords)
            }
          }
      }
     saveTopicWords("lda-featured-topicWords-tst.csv", topicWords)

 } 
  
    def computeKMeans(
      ldaParagraphs: org.apache.spark.sql.DataFrame, 
      nbClusters:Int ) = {
     

        
      var nbIterations = 1000
      var initSteps = 2
      var seed = new org.apache.spark.ml.clustering.KMeans().getSeed
      var scaling = false
      var prop = System.getProperty("rcp216.lda.kmeans.nbIterations")
      if ( prop != null ) {
        nbIterations = prop.toInt
      }
      prop = System.getProperty("rcp216.lda.kmeans.initSteps")
      if ( prop != null ) {
        initSteps = prop.toInt
      }
      prop = System.getProperty("rcp216.lda.kmeans.seed")
      if ( prop != null ) {
        seed = prop.toLong
      }
      val kmeans = new org.apache.spark.ml.clustering.KMeans()
        .setK( nbClusters )
        .setMaxIter( nbIterations)
        .setInitSteps( initSteps )
        .setFeaturesCol( "topicDistribution" )
        .setSeed( seed )
        .setInitMode("k-means||")
        .setPredictionCol("featureTopic")
      
      val kmeansModel = kmeans.fit( ldaParagraphs )
      val kmeanParagraphs = kmeansModel.transform( ldaParagraphs )
      ( kmeanParagraphs, kmeansModel)
    } 
    

  def saveMeasures( path: String, measures : Array[ (Int, Double, Double, Double, Double, Double) ]) = {
    val ps = new java.io.PrintStream(new java.io.FileOutputStream(path))
    ps.println("topic\tlogLikehood\tlogPerplexity\twsse\tword2vect\tUMass")
    measures.foreach( row => ps.println( row._1 + "\t" + row._2 + "\t" + row._3 + "\t" + row._4 + "\t" + row._5 + "\t" + row._6))
    ps.close()
  }
  

}