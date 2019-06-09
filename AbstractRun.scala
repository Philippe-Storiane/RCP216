package com.rcp216.racineTopic

abstract class AbstractRun extends Serializable{
  
   def searchClusterSize(minCluster: Int, maxCluster:Int, sc:org.apache.spark.SparkContext, spark: org.apache.spark.sql.SparkSession) 
  
   def analyzeCluster( ocDF: org.apache.spark.sql.DataFrame, nbClusters: Int)
   
   def saveTopicMap( path: String, topicMap: Array[ ( Long, Array[( Int, Double)])]) = {
   val ps = new java.io.PrintStream(new java.io.FileOutputStream(path))
   val topicLength = topicMap(0)._2.length
    ps.print("id")
    for( index <- 1 to topicLength) {
      ps.print("\ttopic_index_"+ index + "\ttopic_weight_" + index)
    }
    ps.println()
    topicMap.foreach( row => {
      ps.print( row._1 )
      for( index <- 0 to (topicLength - 1)) {
        ps.print("\t" + row._2(index)._1 + "\t" + row._2( index )._2)
      }
      ps.println()
      }    
    )
    ps.close()    
  }
  
  def saveTopicWords(
      path:String,
      topicWords: Array[ ( Int, scala.collection.mutable.WrappedArray[ (String, Double)])]) = {
    val ps = new java.io.PrintStream(new java.io.FileOutputStream(path))
    val nbTerms = topicWords(0)._2.length
    ps.print("topic_id")
    for( index <- 1 to nbTerms) {
      ps.print("\tterm_name_"+ index + "\tterm_weight_" + index)
    }
    ps.println()
    topicWords.foreach( row => {
      val topic_id = row._1
      ps.print( topic_id )
      for( index <- 0 to (nbTerms - 1)) {
        val term_name = row._2( index)._1 
        val term_weight = row._2( index)._2
        ps.print("\t\"" + term_name + "\"\t" + term_weight)
      }
      ps.println()
      }    
    )
    ps.close()     
  }
  
}