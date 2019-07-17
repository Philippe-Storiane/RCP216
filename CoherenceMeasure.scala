package com.rcp216.racineTopic

object CoherenceMeasure {
  
  val ETA = 1.0 / 500
  
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
    val result = topicTopWords.map( topic => {
        val topicIndex = topic._1
        val termIndices = topic._2
        val wordsLength = topic._2.length
        var topicMeasure = 0.0
        for( i <- 0 to wordsLength - 1) {
          val wordIndex_i = termIndices(i)
          val wordText_i = vocab( wordIndex_i )
          if ( bWordEmbeddings.value.get( wordText_i) != None ) {
            val wordEmbeddings_i = bWordEmbeddings.value( wordText_i )
            val word_i = new breeze.linalg.DenseVector[ Double ] ( wordEmbeddings_i.toArray)
            val word_i_norm = breeze.linalg.norm( word_i)
            for(j <- i + 1  to wordsLength - 1) {
              val wordIndex_j = termIndices(j)
              val wordText_j = vocab( wordIndex_j )
              if ( bWordEmbeddings.value.get(wordText_j ) != None ) {
                val wordEmbeddings_j = bWordEmbeddings.value( wordText_j )
                val word_j = new breeze.linalg.DenseVector[ Double ] ( wordEmbeddings_j.toArray)
                topicMeasure = topicMeasure + ( word_i dot word_j ) /( word_i_norm * breeze.linalg.norm( word_j))
              } else {
                println("unknown word " + wordText_j + " for topic " + topicIndex)
              }
            }
          } else {
            println("unknown word " + wordText_i + " for topic " + topicIndex)
          }          
        }
        ( topicIndex, topicMeasure / wordsLength)
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
    for( ( r,c) <- wocCounts.activeKeysIterator) {
      val denom = wocCounts(c, c)
      val joint = wocCounts(r, c) / denom
      wocCounts(r, c) = joint
    }
    wocCounts

  }
  
  def uMass( topicTopWords: Array[ ( Int, Array[Int] ) ], corpusPMI: breeze.linalg.CSCMatrix[Double] )= {
    var topicNumber = topicTopWords.length
    val result = topicTopWords.map( topic => {
        val topicIndex = topic._1
        val termIndices = topic._2
        val wordsLength = topic._2.length
        var measure = 0.0
        for ( List(w1,w2) <- termIndices.toList.combinations(2)) {
          val pw1_w2 = java.lang.Math.log( corpusPMI( w1,w2) + ETA )
          val pw1 = java.lang.Math.log( corpusPMI(w1, w1) )
          val pw2 = java.lang.Math.log( corpusPMI(w2,w2) )
          measure = measure + 2 * pw1_w2 - pw1 - pw2 
        }
        ( topicIndex, measure / wordsLength )
      }
    )
    result
  }
}