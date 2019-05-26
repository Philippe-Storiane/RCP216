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
  
  def run( nbClusters:Int, top:Int, sc:org.apache.spark.SparkContext, spark: org.apache.spark.sql.SparkSession) = {
    val contentExtractor = new ContentExtractor()
    var paragraphs = contentExtractor.extractContent(sc)
    var ( paragraphsDF, vocab )  = contentExtractor.extractDataFrame( paragraphs, sc, spark)
    val sentence = paragraphsDF.select("filtered")
    
     val word2vec = new org.apache.spark.ml.feature.Word2Vec().setInputCol("filtered")
     
    
    val word2vecModel = word2vec.fit( sentence )
    val wordEmbeddings2 = word2vecModel.getVectors.collect().map( row => ( row.getString(0), row.getAs[org.apache.spark.ml.linalg.DenseVector](1))).toMap
    
    
    
    
    val wordEmbeddings = sc.textFile("frWac_non_lem_no_postag_no_phrase_200_skip_cut100.txt").map(
        line => {
          val parts = line.split("[\t,]")
          val word = parts(0)
          val data = parts.slice(1, parts.length).map(_.toDouble)
          ( word, org.apache.spark.ml.linalg.Vectors.dense( data ) )
        }
    ).collectAsMap()
    val bWordEmbeddings = sc.broadcast( wordEmbeddings )
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
     
    
   val nbIterations = 100
    val kmeans = new org.apache.spark.ml.clustering.KMeans()
      .setK( nbClusters )
      .setMaxIter( nbIterations)
      .setFeaturesCol("value")
      .setInitMode("k-means||")
      .setPredictionCol("topic")
      
    val kmeansModel = kmeans.fit( sentence2vec )
    val kmeanParagraphs = kmeansModel.transform( sentence2vec )
    
    val WSSSE = kmeansModel.computeCost( kmeanParagraphs )
    println(s"Within Set Sum of Squared Errors = $WSSSE")
    
    kmeansModel.clusterCenters.foreach( cluster => {
      println("-")
       word2vecModel.findSynonyms(cluster, top)
         .foreach(synonym => print(
           " %s (%5.3f),"
           .format(
               synonym.getString(0),
               synonym.getDouble(1))
          )
       )
      }
    )
    
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
}