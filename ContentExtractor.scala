package com.rcp216.racineTopic


import org.apache.spark.sql.DataFrame


import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.IDF
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.ml.feature.CountVectorizerModel
import org.apache.spark.ml.linalg.SparseVector



import scala.collection.JavaConversions.asScalaBuffer
import scala.collection.mutable.ArrayBuffer


import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.SingularValueDecomposition
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.{ SparseVector => MLibSparseVector }
import org.apache.spark.mllib.linalg.{ Vectors => MLibVectors }

import breeze.linalg.{ DenseVector => BDV, SparseVector => BSV, Vector => BV }


class ContentExtractor extends Serializable {
  


  def sparseSparkToBreeze( sparkSparse: org.apache.spark.ml.linalg.SparseVector) = {
    sparkSparse match {
      case org.apache.spark.ml.linalg.SparseVector(size, indices, values) => new breeze.linalg.SparseVector[Double](indices, values, size)
    }
  }
  
  def sparseBreezeToSpark( breezeSparse: breeze.linalg.SparseVector[Double]) = {
    breezeSparse.array.compact
    org.apache.spark.ml.linalg.Vectors.sparse(breezeSparse.size, breezeSparse.array.index, breezeSparse.array.data)
  }
  
  
  def denseSparkToBreeze( sparkDense: org.apache.spark.ml.linalg.DenseVector) = {
    new breeze.linalg.DenseVector[Double](sparkDense.toArray)
  }
  
  def denseBreezeToSpark( breezeVector: breeze.linalg.DenseVector[ Double ]) = {
    new org.apache.spark.ml.linalg.DenseVector(breezeVector.toArray)
  }


  def extractParagraphs( play: IndexedSeq[String]) = {
    val playLength = play.length - 1
    var paragraph = Array("")
    paragraph = paragraph.patch(0, Nil, 1)
    var displayParagraph = false
    val paragraphs = for( index <- 0 to playLength
        if {
          if ( displayParagraph ) {
            paragraph = Array("")
            paragraph = paragraph.patch(0, Nil, 1)
            paragraph = paragraph.patch(0, Nil, 1)
            displayParagraph = false
          }
          val tokens =  play( index).split( " ")
            .filter( s => s.length > 2  )
          val isParagraphSeparator = tokens.fold( false){ case (a:Boolean, b:String) => ( a || (b.toUpperCase() == b))}
          if ( tokens.isEmpty || isParagraphSeparator.asInstanceOf[ Boolean] ) {
            if ( ! paragraph.isEmpty) displayParagraph = true
          } else {
            paragraph ++= tokens
          }
          displayParagraph
        }

    ) yield paragraph
    paragraphs
  }
  
    
  
     


  def extractContent(sc:org.apache.spark.SparkContext) = {
    val content = sc.textFile("standford-nlp.txt").collect()
    val paragraphs = extractParagraphs( content )
    paragraphs
    //mythridateParagraphs
  }
    
  def extractRDD( paragraphs: IndexedSeq[Array[String]], sc:org.apache.spark.SparkContext) = {
    val stopWords = loadStopWords( sc )
    val content = for( index <-0 to paragraphs.length - 1 ) yield {
      val filtered = paragraphs(index).filterNot( word => stopWords.contains( word ))
      ( index.toString, filtered)
    }
    var doc = sc.parallelize( content )
    doc.filter( row => row._2.size > 0)
  }

  def loadStopWords( sc: org.apache.spark.SparkContext ) = {
    val stopWords = org.apache.spark.ml.feature.StopWordsRemover.loadDefaultStopWords("french")
    val additionalWords = sc.textFile("additional-stop-words.txt").collect()
    stopWords union additionalWords
    
  }
  
   
  def extractDataFrame( content: IndexedSeq[Array[String]], sc:org.apache.spark.SparkContext, spark: org.apache.spark.sql.SparkSession) = {
    
   
    val paragraphsDF = spark.createDataset( content )(org.apache.spark.sql.catalyst.encoders.ExpressionEncoder(): org.apache.spark.sql.Encoder[Array[String]])toDF("rawText")



    //
    // remove stop words
    //
    val remover = new org.apache.spark.ml.feature.StopWordsRemover()
      .setStopWords( loadStopWords( sc ))
      .setInputCol("rawText")
      .setOutputCol("filtered")

    val filteredParagraphs = remover.transform(paragraphsDF).filter( row => {
        val filtered = row.getAs[scala.collection.mutable.WrappedArray[ String ] ]("filtered")
        filtered.size > 0
      }
    )
    
    //
    // TF IDF
    //
    val minDFProp = System.getProperty("rcp216.minDF")
    var minDF = 2
    if ( minDFProp != null) {
      minDF = minDFProp.toInt
    }
    val countVectorizerModel = new org.apache.spark.ml.feature.CountVectorizer()
      .setInputCol("filtered")
      .setOutputCol("tf")
      .setMinDF( minDF )
      .fit( filteredParagraphs )
    

    val termFrequencyParagraphs = countVectorizerModel.transform( filteredParagraphs)
    
    val idf = new org.apache.spark.ml.feature.IDF().setInputCol("tf").setOutputCol( "idf")
    
    val idfModel = idf.fit( termFrequencyParagraphs)
    val paragraphFeatures = idfModel.transform( termFrequencyParagraphs )
    val minNonzeros = getMinNonzeros()
    val filteredParagraphFeatures = paragraphFeatures.filter( row => {
        val tf = row.getAs[ org.apache.spark.ml.linalg.SparseVector ]("tf")
        tf.numNonzeros > minNonzeros
      }  
    )
    ( filteredParagraphFeatures, countVectorizerModel.vocabulary)
  }
  
  def getMinNonzeros():Int = {
    var minNonzeros = 0
    var prop = System.getProperty("rcp216.minNonzeros")
    if ( prop != null ) {
      minNonzeros = prop.toInt
    }
    minNonzeros
  }
}