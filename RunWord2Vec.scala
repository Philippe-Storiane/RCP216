package com.rcp216.racineTopic

import org.apache.spark.SparkContext

import org.apache.spark.sql.Encoder
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.{Word2Vec, Word2VecModel}
import org.apache.spark.ml.linalg.{ DenseVector => SparkDenseVector}


import breeze.linalg.{squaredDistance, DenseVector => BreezeDenseVector, Vector}

object RunWord2Vec {
  
  def searchClusterSize( minCluster: Int, maxCluster:Int, sc:org.apache.spark.SparkContext, spark: org.apache.spark.sql.SparkSession) = {
    val contentExtractor = new ContentExtractor()
    var paragraphs = contentExtractor.extractContent(sc)
    var ( paragraphsDF, vocab )  = contentExtractor.extractDataFrame( paragraphs, sc, spark)
    val sentence = paragraphsDF.select("filtered")
    val sentence2vec = extractWord2Vec( sentence, sc, spark)    
    val wsse = for ( nbClusters <- minCluster to maxCluster ) yield {
      println("Computing Kmeans for " + nbClusters + " clusters")
      val ( kmeansParagraphs, kmeansModel ) = computeKMeans(  sentence2vec, nbClusters,  sc, spark )
      ( nbClusters, kmeansModel.computeCost(kmeansParagraphs))
    }
    saveWSSE("word2vec-wsse.csv", wsse.toArray)
  }
  
  def analyzeCluster( nbCluster:Int, sc:org.apache.spark.SparkContext, spark: org.apache.spark.sql.SparkSession) = {
    val contentExtractor = new ContentExtractor()
    var paragraphs = contentExtractor.extractContent(sc)
    var ( paragraphsDF, vocab )  = contentExtractor.extractDataFrame( paragraphs, sc, spark)
    val sentence = paragraphsDF.select("filtered")
    val sentence2vec = extractWord2Vec( sentence, sc, spark)
    val ( kmeansParagraphs, kmeansModel ) = computeKMeans(  sentence2vec, nbCluster, sc, spark )
    val bWordEmbeddings = CoherenceMeasure.loadWordEmbeddings( sc )
    findSynonyms(kmeansModel, 10, vocab, bWordEmbeddings)
  }
  
  
  def extractWord2Vec(
      sentence: org.apache.spark.sql.DataFrame, 
      sc:org.apache.spark.SparkContext,
      spark: org.apache.spark.sql.SparkSession) = {
    val bWordEmbeddings = CoherenceMeasure.loadWordEmbeddings( sc )
    val contentExtractor = new ContentExtractor()
    val vocabSize = 200
    val sentence2vec = sentence.map( row => {
        val text = row.getAs[scala.collection.mutable.WrappedArray[ String ]]("filtered")
        var sentenceVector = breeze.linalg.DenseVector.zeros[ Double ]( vocabSize )
        var wordNumber = 0
        
        text.foreach( word => {
            var value = bWordEmbeddings.value.get( word )
            if ( value != None ) {
              val wordEmbedding  = bWordEmbeddings.value( word )
              sentenceVector += contentExtractor.denseSparkToBreeze( wordEmbedding.asInstanceOf[ org.apache.spark.ml.linalg.DenseVector ] )
              wordNumber += 1
            }
          }
        )

        if ( wordNumber != 0 ) {
          sentenceVector = sentenceVector * ( 1.0 / wordNumber )
        }

        
        contentExtractor.denseBreezeToSpark( sentenceVector ).asInstanceOf[ org.apache.spark.ml.linalg.Vector ]

      }
    )(org.apache.spark.sql.catalyst.encoders.ExpressionEncoder(): org.apache.spark.sql.Encoder[ org.apache.spark.ml.linalg.Vector ])
    sentence2vec

  }
  
  def computeKMeans(
      sentence2vec: org.apache.spark.sql.Dataset[org.apache.spark.ml.linalg.Vector], 
      nbClusters:Int,
      sc:org.apache.spark.SparkContext,
      spark: org.apache.spark.sql.SparkSession ) = {
     
    // val word2vec = new org.apache.spark.ml.feature.Word2Vec().setInputCol("filtered")
     
    
    // val word2vecModel = word2vec.fit( sentence )
    // val wordEmbeddings2 = word2vecModel.getVectors.collect().map( row => ( row.getString(0), row.getAs[org.apache.spark.ml.linalg.DenseVector](1))).toMap
    
    
    

        
    val nbIterations = 50
    val kmeans = new org.apache.spark.ml.clustering.KMeans()
      .setK( nbClusters )
      .setMaxIter( nbIterations)
      .setFeaturesCol("value")
      .setInitMode("k-means||")
      .setPredictionCol("topic")
      
      val kmeansModel = kmeans.fit( sentence2vec )
      val kmeanParagraphs = kmeansModel.transform( sentence2vec )
 //     val l = findSynonyms(kmeansModel, top,  vocab, bWordEmbeddings )

    //saveWSSE( "word2vec-wsse.csv", wsse.toArray)
      ( kmeanParagraphs, kmeansModel)
  }
  
  def findSynonyms( kmeansModel: org.apache.spark.ml.clustering.KMeansModel,
      top:Int, vocab: Array[ String ],
      bWordEmbeddings: org.apache.spark.broadcast.Broadcast[scala.collection.Map[String, org.apache.spark.ml.linalg.Vector]]) = {
    val realVocab = vocab.filter( word => bWordEmbeddings.value.get( word ) != None)
    kmeansModel.clusterCenters.foreach(  cluster =>
      {
        println("")
        val bCluster = breeze.linalg.DenseVector( cluster.toArray )
        val bClusterLength = breeze.linalg.norm( bCluster )
        // val ordering = Ordering.by[(String, Double( data => data._2)
        val words = realVocab.map( word =>
          {
            val vectr =  bWordEmbeddings.value.get( word )
            val bVectr = new breeze.linalg.DenseVector[ Double ] (bWordEmbeddings.value( word ).toArray)
            val dist = ( bCluster dot bVectr ) / ( bClusterLength * breeze.linalg.norm( bVectr ))
            ( word, dist )  
          }
        )
        val synonyms = words.sortWith( _._2 > _._2).take(top)
         .foreach(synonym => print(
           " %s (%5.3f),"
           .format(
               synonym._1,
               synonym._2)
          )      
        )
      }
    )
  }
  
      
  def saveWSSE( path: String, wsse : Array[ (Int, Double) ]) = {
    val ps = new java.io.PrintStream(new java.io.FileOutputStream(path))
    wsse.foreach( cost => ps.println(cost._1 + "\t"+ + cost._2 ))
    ps.close()
  }
}