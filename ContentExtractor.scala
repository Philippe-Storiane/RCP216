package com.rcp216.racineTopic


import org.apache.spark.sql.DataFrame


import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.IDF
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.ml.feature.CountVectorizerModel
import org.apache.spark.ml.linalg.SparseVector



import fllemmatizer.FLLemmatizer
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


  def extractPlay( text:Array[String], beginTag: String, endTag: String) = {
    var inPlay = false
    var line = ""
    val textLength = text.length - 1
    val play =
      for( index <- 0 to textLength 
        if {
          line = text( index )
          if ( inPlay ) {
            if ( line.indexOf(endTag) != -1  ) {
              inPlay = false
            }
          } else {
            if ( line.indexOf(beginTag) != -1 ) {
              inPlay = true
            }
          }
          inPlay
        }
      ) yield line
    play
  }

  def extractParagraphs( play: IndexedSeq[String]) = {
    val playLength = play.length - 1
    var paragraph = Array("")
    paragraph = paragraph.patch(0, Nil, 1)
    val lemmatizer = new fllemmatizer.FLLemmatizer("fr")
    var displayParagraph = false
    val paragraphs = for( index <- 0 to playLength
        if {
          if ( displayParagraph ) {
            paragraph = Array("")
            paragraph = paragraph.patch(0, Nil, 1)
            paragraph = paragraph.patch(0, Nil, 1)
            displayParagraph = false
          }
          val tokens = scala.collection.JavaConversions.asScalaBuffer( lemmatizer.lemmatize( play( index), true))
            .filter( s => ( s(1) != "PUNC") && ( s(1) != "NUM") && ( s(2).length > 2 ) )
          val isParagraphSeparator = tokens.fold( false){ case (a:Boolean, b:Array[String]) => ( a || (b(0).toUpperCase() == b(0)))}
          if ( tokens.isEmpty || isParagraphSeparator.asInstanceOf[ Boolean] ) {
            if ( ! paragraph.isEmpty) displayParagraph = true
          } else {
            val words = tokens.map( s => s(2))
            paragraph ++= words
          }
          displayParagraph
        }

    ) yield paragraph
    paragraphs
  }
  
    
  
  def extractVerses( play: IndexedSeq[ String]) = {
    val lemmatizer = new fllemmatizer.FLLemmatizer("fr")
    val playLength = play.length - 1
    var verse = Seq("")
    verse = verse.patch(0, Nil, 1)
    var displayVerse = false

    val verses = for( index <- 0 to playLength
        if {
          if ( displayVerse ) {
            verse = Seq("")
            verse = verse.patch(0, Nil, 1)
            displayVerse = false
          }

          val tokens = scala.collection.JavaConversions.asScalaBuffer(lemmatizer.lemmatize(play( index), true))
              .filter( s => ( s(1) != "PUNC") && ( s(1) != "NUM") && ( s(2).length > 2 ) )
          val isParagraphSeparator = tokens.fold( false){ case (a:Boolean, b:Array[String]) => ( a || (b(0).toUpperCase() == b(0)))}
          if (  tokens.isEmpty ||  isParagraphSeparator.asInstanceOf[ Boolean] ) {
            displayVerse = false
          } else {
            verse = tokens.map(s => s(2))
            displayVerse = true
          }
          displayVerse
        }

    ) yield verse
    verses
  }
  
  
  def extractPOS( play: IndexedSeq[ String]) = {
    val lemmatizer = new fllemmatizer.FLLemmatizer("fr")
    val playLength = play.length - 1
    play.map( line => lemmatizer.lemmatize( line ))
  }
 
/*
    def extractSentences( play: IndexedSeq[ String ]) = {
  val playLength = play.length - 1
  val lemmatizer = new FLLemmatizer("fr")

  var displaySentence = false
  
  var sentence = Seq("")
  sentence = sentence.patch(0,Nil, 1)
  var paragraph = Array( sentence )
  paragraph = paragraph.patch(0, Nil, 1)
  val estherSentences = for( index <- 0 to playLength
      if {
        println("index " + index)
        if ( displaySentence ) {
          
          sentence = Seq("")
          sentence = sentence.patch(0,Nil, 1)
          paragraph = Array( sentence)
          paragraph = paragraph.patch(0, Nil, 1)
          displaySentence = false
        }
        val tokens = lemmatizer.lemmatize( play( index), true)
            .filter( s => (  s(1) != "NUM") && ( ( s(2).length > 1 ) || ( s(1) == "PUNC")) )
        val isParagraphSeparator = tokens.fold( false){ case (a:Boolean, b:Array[String]) => ( a || ( (b(1) != "PUNC" ) && ( b(0).toUpperCase() == b(0))))}
        displaySentence = true
        if (  tokens.isEmpty ||  isParagraphSeparator.asInstanceOf[ Boolean] ) {
          println("In Separator" + play( index))
          if( paragraph.length > 1 ) {
            if (  ! sentence.isEmpty ) paragraph ++= Array(sentence)
            displaySentence = true
          }
          println( "paragraph" + paragraph.length)
        } else {
          println( "separator" + isParagraphSeparator.asInstanceOf[ Boolean])
          for (token <- 0 to tokens.length - 1) {
              if ( tokens( token )(1) == "PUNC") {
                if ( (tokens( token )(0) == ".") || ( tokens( token)(0) == ";") ) 
                  if (  ! sentence.isEmpty  ) {
                    paragraph ++= Array(sentence)
                    sentence = Seq("")
                    sentence = sentence.patch(0,Nil, 1)
                  }
              } else {
                sentence ++= Seq(tokens( token )(2))
              }
          }
        }
        true //displaySentence
      }

  ) yield paragraph
  estherSentences
    }
*/
    //
    // extract data
    //
    


  def extractContent(sc:org.apache.spark.SparkContext) = {
    val estherText = sc.textFile("esther-utf8.txt").collect()
    val estherPlay = extractPlay( estherText, "ACTE PREMIER.", "NOTES TO PROLOGUE.")
    val estherParagraphs = extractParagraphs( estherPlay )
    val mythridateText = sc.textFile("mythridate-utf8.txt").collect()
    val mythridatePlay = extractPlay( mythridateText, "ACTE PREMIER", "End of the Project Gutenberg")
    val mythridateParagraphs = extractParagraphs( mythridatePlay )
    
    //val mythridateParagraphs = extractVerses( mythridatePlay )
    estherParagraphs ++ mythridateParagraphs
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
    org.apache.spark.ml.feature.StopWordsRemover.loadDefaultStopWords("french")
  }
  
   
  def extractDataFrame( content: IndexedSeq[Array[String]], sc:org.apache.spark.SparkContext, spark: org.apache.spark.sql.SparkSession) = {
    
   
    val paragraphsDF = spark.createDataset( content )(org.apache.spark.sql.catalyst.encoders.ExpressionEncoder(): org.apache.spark.sql.Encoder[Array[String]])toDF("rawText")



    //
    // remove stop words
    //
    val remover = new org.apache.spark.ml.feature.StopWordsRemover()
      .setStopWords( org.apache.spark.ml.feature.StopWordsRemover.loadDefaultStopWords("french"))
      .setInputCol("rawText")
      .setOutputCol("filtered")

    val filteredParagraphs = remover.transform(paragraphsDF)
    
    //
    // TF IDF
    //
    val countVectorizerModel = new org.apache.spark.ml.feature.CountVectorizer()
      .setInputCol("filtered")
      .setOutputCol("tf")
      .setMinDF(2)
      .fit( filteredParagraphs )
      
    val termFrequencyParagraphs = countVectorizerModel.transform( filteredParagraphs)
    
    val idf = new org.apache.spark.ml.feature.IDF().setInputCol("tf").setOutputCol( "idf")
    
    val idfModel = idf.fit( termFrequencyParagraphs)
    val paragraphFeatures = idfModel.transform( termFrequencyParagraphs )

    ( paragraphFeatures, countVectorizerModel.vocabulary)
  }
}