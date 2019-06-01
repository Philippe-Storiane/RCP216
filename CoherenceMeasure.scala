package com.rcp216.racineTopic

object CoherenceMeasure {
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
    
    
    docDF.select("tf").foreach( row => {
        val  termIndices = row.getAs[org.apache.spark.ml.linalg.SparseVector](0)
        val termIds = termIndices.indices
        for ( termId <- termIds) {
          wocCounts( termId, termId) += 1.0
        }
        for ( List(w1,w2) <- termIds.toList.combinations(2)) {
          wocCounts( w1, w2) += 1.0
          wocCounts( w2, w1) += 1.0
        }
      }
    )
    for (r <- 0 until wocCounts.rows;
         c <- r until wocCounts.cols ) {
            val denom = wocCounts(c, c)
            val joint = wocCounts(r, c) / denom
            wocCounts(r, c) = joint
        }
    wocCounts

  }
  
}