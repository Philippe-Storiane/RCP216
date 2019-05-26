package com.rcp216.racineTopic

import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession

import org.apache.spark.ml.clustering.LDA
import org.apache.spark.ml.clustering.DistributedLDAModel

object RunLDA {
  
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
  
}