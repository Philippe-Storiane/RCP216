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
class Test {
  
  def getEnv(
      sc:org.apache.spark.SparkContext,
      spark: org.apache.spark.sql.SparkSession) = {
    val ce = new ContentExtractor()
    val content = ce.extractContent( sc )
    val paragraphsDF = spark.createDataset( content )(org.apache.spark.sql.catalyst.encoders.ExpressionEncoder(): org.apache.spark.sql.Encoder[Array[String]])toDF("rawText")



    //
    // remove stop words
    //
    val remover = new org.apache.spark.ml.feature.StopWordsRemover()
      .setStopWords( ce.loadStopWords( sc ))
      .setInputCol("rawText")
      .setOutputCol("filtered")

    val filteredParagraphs = remover.transform(paragraphsDF)
    
    //
    // TF IDF
    //

    val countVectorizerModel = new org.apache.spark.ml.feature.CountVectorizer()
      .setInputCol("filtered")
      .setOutputCol("tf")
      .setMinDF( 3 )
      .fit( filteredParagraphs )
    val vocab = countVectorizerModel.vocabulary
    val p = countVectorizerModel.transform( filteredParagraphs)
    var index = 0
    p.collect().foreach( row => {

        println("index " + index)
        var filtered = row.getAs[scala.collection.mutable.WrappedArray[String]]("filtered")
        var tfVector = row.getAs[org.apache.spark.ml.linalg.SparseVector]("tf")
        var number = 0.0
        tfVector.indices.foreach( index => {
            val  tf = tfVector.apply(index)
            if ( tf == 0.0 ) {
              println("tf 0 at index " + index)
            }
            val word = vocab( index )
            if ( ! filtered.contains( word )) {
              println(index + "unknown word " + word)
            }
            number += tf
          }
        )
        /*
        if ( ( filtered.size * 1.0 ) != number) {
          println(index + " discrepencies in number " + filtered.size + " " + number + "\n" + "for filter " + filtered)
        }
        */
        index = index + 1
      }
    )
  }
 
}