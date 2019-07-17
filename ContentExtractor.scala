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
    org.apache.spark.mllib.linalg.Vectors.sparse(breezeSparse.size, breezeSparse.array.index, breezeSparse.array.data)
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
    val content = sc.textFile("stanford-nlp.txt").collect()
    val paragraphs = extractParagraphs( content )
    paragraphs
    //mythridateParagraphs
  }
    
  def extractRDD( paragraphs: IndexedSeq[Array[String]], sc:org.apache.spark.SparkContext) = {
    val stopWords = loadStopWords()
    val content = for( index <-0 to paragraphs.length - 1 ) yield {
      val filtered = paragraphs(index).filterNot( word => stopWords.contains( word ))
      ( index.toString, filtered)
    }
    var doc = sc.parallelize( content )
    
    doc
  }

  def loadStopWords() = {
    val stopWords = org.apache.spark.ml.feature.StopWordsRemover.loadDefaultStopWords("french")
    val adverbStopWords = Array(
        "tel",  
        "tout",
        "que?",
        "autre",
        "cher",
        "fois",
        "comment?",
        "jusqu'à",
        "cla",
        "encor",
        "nouveau",
        "combien",
        "moindre",
        "point",
        "pourquoi?",
        "quelque",
        "livrer",
        "prêter",//
        "aujourd'hui",
        "former",
        "seul",//
        "juste",
        "grand",
        "prêt",
        "nouveau",
        "propre",//
        "sain",//
        "propre",
        "hélas",
        "plein",
        "cld",
        "dernier", // TBC
        "rang", // TBC
        "premier")
    val verbalStopWords = Array(
        "voir",
        "tenir",// TBC
        "être",
        "assurer",//
        "avoir",
        "faire",
        "voir",
        "pouvoir",
        "vouloir",
        "entendre",//
        "laisser",//
        "oser",//
        "porter",//
        "courir",//
        "dire",
        "savoir",
        "venir",
        "aller",
        "mettre",
        "reste",
        "prétendre",
        "chercher",
        "suivre",
        "rendre",
        "donner",
        "quitter",
        "recevoir",
        "demander",//
        "trouver",//
        "écouter",
        "montrer",
        "passer")
    val nameStopWords = Array(
        "Mardochée",
        "Monime",
        "Pharnace",
        "Esther",
        "Xipharès",
        "Mithridate",
        "Aman",
        "Phoedime",
        "Sion",
        "Assuérus",
        "Arbate",
        "Asaph",
        "Grèce",
        "Rome")
    val subjectStoWords= Array(
        "Monsieur",
        "nom",
        "main",
        "romain",
        "mot",
        "front",//
        "voix",//
        "soin",//
        "lieux",//
        "heure",// TBC
        "place",//
        "persan",
        "juif",
        "Israel",
        "bouche",
        "moment",//
        "oreille",
        "histoire",//
        "présent", // ambiguité des sens
        "genou", // TBC
        "effet",
        "yeux",
        "sujet", // half style
        "chemin" // half style
        )
    stopWords union adverbStopWords union verbalStopWords union nameStopWords union subjectStoWords
    
  }
  
   
  def extractDataFrame( content: IndexedSeq[Array[String]], sc:org.apache.spark.SparkContext, spark: org.apache.spark.sql.SparkSession) = {
    
   
    val paragraphsDF = spark.createDataset( content )(org.apache.spark.sql.catalyst.encoders.ExpressionEncoder(): org.apache.spark.sql.Encoder[Array[String]])toDF("rawText")



    //
    // remove stop words
    //
    val remover = new org.apache.spark.ml.feature.StopWordsRemover()
      .setStopWords( loadStopWords())
      .setInputCol("rawText")
      .setOutputCol("filtered")

    val filteredParagraphs = remover.transform(paragraphsDF)
    
    //
    // TF IDF
    //
    val countVectorizerModel = new org.apache.spark.ml.feature.CountVectorizer()
      .setInputCol("filtered")
      .setOutputCol("tf")
      .setMinDF(7)
      .fit( filteredParagraphs )
      
    val termFrequencyParagraphs = countVectorizerModel.transform( filteredParagraphs)
    
    val idf = new org.apache.spark.ml.feature.IDF().setInputCol("tf").setOutputCol( "idf")
    
    val idfModel = idf.fit( termFrequencyParagraphs)
    val paragraphFeatures = idfModel.transform( termFrequencyParagraphs )

    ( paragraphFeatures, countVectorizerModel.vocabulary)
  }
}