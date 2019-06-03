package com.rcp216.racineTopic


import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

import org.apache.spark.rdd.RDD

import org.apache.spark.graphx.Edge
import org.apache.spark.graphx.VertexId
import org.apache.spark.graphx.Graph


import scala.collection.mutable.WrappedArray

/*object Test {
  
  
     def extractPlay( text:Array[String], beginTag: String, endTag: String) = {
     var inPlay = true //false
      var line = ""
      val textLength = text.length - 1
      val play =
        for( index <- 0 to textLength 
          if {
            line = text( index )
           inPlay
          }
        ) yield line
      play
     }
  
}
*/
class Test extends Serializable{
     def main( args: Array[String], sc:SparkContext, spark:SparkSession) = {
     var contentExtractor = new ContentExtractor()
     var paragraphs = contentExtractor.extractContent(sc)
     val (docDF, vocabulary) = contentExtractor.extractDataFrame(paragraphs, sc, spark)
      val corpusPMI = CoherenceMeasure.preprocessUMass( docDF, vocabulary.length)
     val nbClusters = 10
     val ( ldaParagraphs, ldaModel ) = RunLDA.computeLDA(docDF, nbClusters)
     val topics = ldaModel.describeTopics(10)
     val topicsDump = topics
        .select("topic","termIndices")
        .rdd
        .map( row => ( row.getAs[Int](0), row.getAs[scala.collection.mutable.WrappedArray[ Int]](1).toSeq.toArray))
        .collect()
     val ldaUMass = CoherenceMeasure.uMass( topicsDump, corpusPMI).map(_._2).sum / nbClusters
        

     }
}