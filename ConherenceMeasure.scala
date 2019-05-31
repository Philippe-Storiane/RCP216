package com.rcp216.racineTopic

object ConherenceMeasure {
  def word2vec(
      topicTopWords: Array[ Array[Int] ],
      vocab: Array[String],
      bWordEmbeddings: org.apache.spark.broadcast.Broadcast[scala.collection.Map[String, org.apache.spark.ml.linalg.Vector]]) = {
    var topicNumber = topicTopWords.length
    var measure = 0.0;
    topicTopWords.foreach( words => {
        val wordsLength = words.length
        var topicMeasure = 0.0
        for( i <- 0 to wordsLength - 1) {
          val wordIndex_i = words(i)
          val wordText_i = vocab( wordIndex_i )
          val wordEmbeddings_i = bWordEmbeddings.value( wordText_i)
          val word_i = new breeze.linalg.DenseVector[ Double ] ( wordEmbeddings_i.toArray)
          val word_i_norm = breeze.linalg.norm( word_i)
          for(j <- i to wordsLength - 1) {
            val wordIndex_j = words(j)
            val wordText_j = vocab( wordIndex_j )
            val wordEmbeddings_j = bWordEmbeddings.value( wordText_j )
            val word_j = new breeze.linalg.DenseVector[ Double ] ( wordEmbeddings_j.toArray)
            topicMeasure = topicMeasure + ( word_i dot word_j ) /( word_i_norm * breeze.linalg.norm( word_j))
          }
        }
        measure = measure + ( topicMeasure / wordsLength )
      }
    )
    measure = measure / topicNumber
  }
  /*
  def uMass( topicTopWords:  Array[ Array[String] ], docDF: org.apache.spark.sql.DataFrame) = {
    var topicNumber = topicTopWords.length
    var measure = 0.0;
    topicTopWords.foreach( words => {
      val wordsLength = words.length
        var topicMeasure = 0.0
        for( i <- 0 to wordsLength - 1) {
          val word_i = docDF.filter{
            match ( Int, Array[Int], Array[Double]) =>
              
          }
          for(j <- i to wordsLength - 1) {
            topicMeasure = 0.0
          }
        }
      }
    )
  }
  */
}