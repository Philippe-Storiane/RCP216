package com.rcp216.racineTopic

object CoherenceMeasure {
  
  
  def loadWordEmbeddings( sc: org.apache.spark.SparkContext) = {
    val wordEmbeddingsFile = "frWac_no_postag_no_phrase_500_skip_cut100.txt"
    // val wordEmbeddingsFile = "frWac_non_lem_no_postag_no_phrase_200_skip_cut100.txt"
    val wordEmbeddings = sc.textFile( wordEmbeddingsFile ).map(
        line => {
          val parts = line.split("[\t,]")
          val word = parts(0)
          val data = parts.slice(1, parts.length).map(_.toDouble)
          ( word, org.apache.spark.ml.linalg.Vectors.dense( data ) )
        }
    ).collectAsMap()
    sc.broadcast( wordEmbeddings )
  }
  
  def word2vec(
      topicTopWords: Array[ ( Int, Array[Int] ) ],
      vocab: Array[String],
      bWordEmbeddings: org.apache.spark.broadcast.Broadcast[scala.collection.Map[String, org.apache.spark.ml.linalg.Vector]]) = {
    var topicNumber = topicTopWords.length
    val combination_top_word = Range(0,10).combinations(2).size
    val result = topicTopWords.map( topic => {
        val topicIndex = topic._1
        val termIndices = topic._2
        val wordsLength = topic._2.length
        var topicMeasure = 0.0
        for( j <- 1 to wordsLength - 1) {
          val wordIndex_j = termIndices(j)
          val wordText_j = vocab( wordIndex_j )
          if ( bWordEmbeddings.value.get( wordText_j) != None ) {
            val wordEmbeddings_i = bWordEmbeddings.value( wordText_j )
            val word_j = new breeze.linalg.DenseVector[ Double ] ( wordEmbeddings_i.toArray)
            val word_j_norm = breeze.linalg.norm( word_j)
            for(i <- 0  to j - 1) {
              val wordIndex_i = termIndices(i)
              val wordText_i = vocab( wordIndex_i )
              if ( bWordEmbeddings.value.get(wordText_i ) != None ) {
                val wordEmbeddings_i = bWordEmbeddings.value( wordText_i )
                val word_i = new breeze.linalg.DenseVector[ Double ] ( wordEmbeddings_i.toArray)
                topicMeasure = topicMeasure + ( word_i dot word_j ) /( word_j_norm * breeze.linalg.norm( word_i))
              } else {
                println("unknown word " + wordText_i + " for topic " + topicIndex)
              }
            }
          } else {
            println("unknown word " + wordText_j + " for topic " + topicIndex)
          }          
        }
        ( topicIndex, topicMeasure / ( wordsLength * combination_top_word))
      }
    )
    result
  }
  
  
  
  def preprocessUMass( docDF: org.apache.spark.sql.DataFrame, vocabSize: Int) = {

    val wocCounts: breeze.linalg.CSCMatrix[Double] = breeze.linalg.CSCMatrix.zeros[Double]( vocabSize, vocabSize )
    
    var index = 0
    docDF.select("tf").collect().foreach( row => {
        println("row No " + index)
        val  termIndices = row.getAs[org.apache.spark.ml.linalg.SparseVector](0)
        val termIds = termIndices.indices
        for ( termId <- termIds) {
          //println("ident " +termId)
          //wocCounts( termId, termId) += 1.0
          wocCounts( termId, termId) += 1.0
          //println(wocCounts( termId, termId))
        }
        for ( List(w1,w2) <- termIds.toList.combinations(2)) {
          //println("w1,w2 ", w1,w2)
          wocCounts( w1, w2) += 1.0
          wocCounts( w2, w1) += 1.0
          //println(wocCounts( w1, w2))
        }
        index += 1
      }
    )
    /*
    for( ( r,c) <- wocCounts.activeKeysIterator) {
      if ( r != c) {
        val denom = wocCounts(c, c)
        val joint = ( wocCounts(r, c)  + ETA )/ denom
        wocCounts(r, c) = joint
      }      
    }
    */
    wocCounts

  }
  
  def uMass( topicTopWords: Array[ ( Int, Array[Int] ) ], corpusPMI: breeze.linalg.CSCMatrix[Double] )= {
    val topicNumber = topicTopWords.length
    val vocabSize = java.lang.Math.sqrt( corpusPMI.size)
    val combination_top_word = Range(0,10).combinations(2).size
    var eta = 1.0
    var prop = System.getProperty("rcp216.umass.eta")
    if ( prop != null ) {
      eta = prop.toDouble
    }
    val result = topicTopWords.map( topic => {
        val topicIndex = topic._1
        val termIndices = topic._2
        val wordsLength = topic._2.length
        var measure = 0.0        
        for ( j <- Range(1, wordsLength)) {
          val wj = termIndices( j )
          for( i <- Range(0, j - 1)) {
            val wi = termIndices( i )
            val pwi_j = corpusPMI(wi, wj) + eta
            val pwj= corpusPMI( wj, wj)
            measure = measure + java.lang.Math.log( pwi_j) - java.lang.Math.log( pwj )
          }
        }
        ( topicIndex, measure / ( wordsLength * combination_top_word ))
      }
    )
    result
  }
}