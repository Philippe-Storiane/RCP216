package com.cnam.rcp216.racineTopic

import org.apache.spark.SparkContext

import org.apache.spark.sql.Encoder
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.feature.StandardScalerModel
import org.apache.spark.ml.feature.{Word2Vec, Word2VecModel}
import org.apache.spark.ml.linalg.{ DenseVector => SparkDenseVector}


import breeze.linalg.{squaredDistance, DenseVector => BreezeDenseVector, Vector}

class RunWord2Vec extends AbstractRun {
  
  def searchClusterSize( minCluster: Int, maxCluster:Int, sc:org.apache.spark.SparkContext, spark: org.apache.spark.sql.SparkSession) = {
    val contentExtractor = new ContentExtractor()
    var paragraphs = contentExtractor.extractContent(sc)
    var ( paragraphsDF, vocab )  = contentExtractor.extractDataFrame( paragraphs, sc, spark)
    val sentence = paragraphsDF.select("filtered", "tf","idf")
    val bWordEmbeddings = CoherenceMeasure.loadWordEmbeddings(sc)
    val sentence2vec = extractWord2Vec( sentence, vocab, bWordEmbeddings, sc, spark)
    val corpusPMI = CoherenceMeasure.preprocessUMass( paragraphsDF, vocab.length )
    val measures = for ( nbClusters <- minCluster to maxCluster ) yield {
      println("Computing Kmeans for " + nbClusters + " clusters")
      val ( kmeansParagraphs, kmeansModel ) = computeKMeans(  sentence2vec, nbClusters,  sc, spark )
      val topicWord = findTopicWords( kmeansModel, 10, vocab, bWordEmbeddings)
      val topicsDump = topicWord.map{
        case ( topicIndex, words) => {
          val wordIndex = words.map( _._1).toArray
          ( topicIndex, wordIndex)
        }
      }
      val wsse = kmeansModel.computeCost(kmeansParagraphs)
      val topicsWord2vec = CoherenceMeasure.word2vec(topicsDump, vocab, bWordEmbeddings)
      val word2vec = topicsWord2vec.map( _._2).sum / nbClusters
      val uMass = CoherenceMeasure.uMass(topicsDump, corpusPMI).map(_._2).sum / nbClusters
      ( nbClusters, wsse, word2vec, uMass)
    }
    saveMeasures("word2vec-measures-tst.csv", measures.toArray)
  }
  
  def analyzeCluster( nbCluster:Int, sc:org.apache.spark.SparkContext, spark: org.apache.spark.sql.SparkSession) = {
    val contentExtractor = new ContentExtractor()
    var paragraphs = contentExtractor.extractContent(sc)
    var ( paragraphsDF, vocab )  = contentExtractor.extractDataFrame( paragraphs, sc, spark)
    val sentence = paragraphsDF.select("filtered", "tf","idf")
    val bWordEmbeddings = CoherenceMeasure.loadWordEmbeddings( sc )
    val sentence2vec = extractWord2Vec( sentence, vocab, bWordEmbeddings, sc, spark)
    val ( kmeansParagraphs, kmeansModel ) = computeKMeans(  sentence2vec, nbCluster, sc, spark )
    
    val topicWords = findTopicWords(kmeansModel, 10, vocab, bWordEmbeddings)
    saveTopicWords("word2vec-topicWords-tst.csv", topicWords)
    findTopicMap( kmeansModel, 5, sentence2vec)
  }
  
  def analyzeRawCluster( nbCluster:Int, sc:org.apache.spark.SparkContext, spark: org.apache.spark.sql.SparkSession) = {
    val contentExtractor = new ContentExtractor()
    var paragraphs = contentExtractor.extractContent(sc)
    var ( paragraphsDF, vocab )  = contentExtractor.extractDataFrame( paragraphs, sc, spark)
    /*
    val tfIdfCompute = org.apache.spark.sql.functions.udf( (tf : org.apache.spark.ml.linalg.SparseVector,
                             idf : org.apache.spark.ml.linalg.SparseVector) => { 
        val bTf = contentExtractor.sparseSparkToBreeze( tf )
        val bIdf = contentExtractor.sparseSparkToBreeze( idf )
        val numActive = bTf.sum
        val btf_idf = ( bTf *:* bIdf ) :*= ( 1.0 / numActive)
        contentExtractor.sparseBreezeToSpark( btf_idf) 
        }
      )
    val paragraphTfIdf = paragraphsDF.withColumn( "tfIdf", tfIdfCompute(org.apache.spark.sql.functions.col("tf"), org.apache.spark.sql.functions.col("tf")))
    */
    val sentence = paragraphsDF.map( row => {
        val tf = row.getAs[org.apache.spark.ml.linalg.SparseVector]("tf")
        val idf = row.getAs[org.apache.spark.ml.linalg.SparseVector]("idf")
        val bTf = contentExtractor.sparseSparkToBreeze( tf )
        val bIdf = contentExtractor.sparseSparkToBreeze( idf )
        val numActive = bTf.sum
        val btf_idf = ( bTf *:* bIdf ) :*= ( 1.0 / numActive)
        contentExtractor.sparseBreezeToSpark( btf_idf) 
      }
    ) (org.apache.spark.sql.catalyst.encoders.ExpressionEncoder(): org.apache.spark.sql.Encoder[ org.apache.spark.ml.linalg.Vector ])
    val ( kmeansParagraphs, kmeansModel ) = computeKMeans(  sentence, nbCluster, sc, spark )
    
    val topicWords = kmeansModel.clusterCenters.zipWithIndex.map{
      case ( clusterCenter, index ) => {
        val arr = clusterCenter.toArray.zipWithIndex.sortBy(-_._1).take(10).map( v => ( v._2, vocab(v._2), v._1))
        val highest:scala.collection.mutable.WrappedArray[(Int, String, Double)] = arr
        ( index, highest)
      }
    }
    saveTopicWords("word2vec-raw-topicWords-tst.csv", topicWords)
    //findTopicMap( kmeansModel, 5, sentence2vec)
  }
  
  def findTopicMap( kmeansModel: org.apache.spark.ml.clustering.KMeansModel, top: Int, sentence2vec:org.apache.spark.sql.Dataset[org.apache.spark.ml.linalg.Vector]) = {
    val isEuclidian:Boolean = isEuclidianDistance()
    val docs = sentence2vec.withColumn("id",org.apache.spark.sql.functions.monotonicallyIncreasingId)
    val topicMap = docs.collect().map( row => {
        val index = row.getAs[ Long ]("id")
        val sentenceVector = row.getAs[ org.apache.spark.ml.linalg.DenseVector ]("value")
        val bSentenceVector = breeze.linalg.DenseVector[Double]( sentenceVector.toArray )
        val bSentenceVectorLength = breeze.linalg.norm ( bSentenceVector )          
        val clustersDist = kmeansModel.clusterCenters.zipWithIndex.map{
          { case ( cluster, index )  => {
              val bCluster = breeze.linalg.DenseVector( cluster.toArray )
              val bClusterLength = breeze.linalg.norm( bCluster )
              var dist = 0.0
              if ( isEuclidian ) {
                dist = org.apache.spark.ml.linalg.Vectors.sqdist( cluster, sentenceVector)
              } else {
                dist = ( bCluster dot bSentenceVector ) / ( bClusterLength * bSentenceVectorLength) 
              }
              ( index, dist )  
            }
          }
        }
        var clustersDistTop = clustersDist
        if ( isEuclidian ) {
          clustersDistTop = clustersDist.sortWith( _._2 < _._2).take( top )
        } else {
          clustersDistTop = clustersDist.sortWith( _._2 > _._2).take( top )
        }
        (index, clustersDistTop)
      }
    )
    saveTopicMap("word2vec-topicMap-tst.csv", topicMap)
    
  }
  
  def extractWord2Vec(
      sentence: org.apache.spark.sql.DataFrame,
      vocab: Array[ String ],
      bWordEmbeddings: org.apache.spark.broadcast.Broadcast[scala.collection.Map[String, org.apache.spark.ml.linalg.Vector]],
      sc:org.apache.spark.SparkContext,
      spark: org.apache.spark.sql.SparkSession) = {
    //val bWordEmbeddings = CoherenceMeasure.loadWordEmbeddings( sc )
    val contentExtractor = new ContentExtractor()
    val vocabSize = 500
    var emptyRow = 0
    val sentence2vec = sentence.map( row => {
        val filtered = row.getAs[scala.collection.mutable.WrappedArray[String]]("filtered")
        val tfVector = row.getAs[ org.apache.spark.ml.linalg.SparseVector]("tf")
        val idfVector = row.getAs[ org.apache.spark.ml.linalg.SparseVector]("idf")
        var sentenceVector = breeze.linalg.DenseVector.zeros[ Double ]( vocabSize )
        var wordWeight = 0.0
        tfVector.indices.foreach( index => {
              val tf = tfVector.apply( index )
              val idf = idfVector.apply( index )
              val word = vocab( index)
              if ( ! filtered.contains( word ) ) {
                println("unknown filtered word " + word)
              }
              val value = bWordEmbeddings.value.get( word ) 
              if ( value != None ) {
                sentenceVector += ( tf * idf ) * contentExtractor.denseSparkToBreeze( bWordEmbeddings.value( word).asInstanceOf[ org.apache.spark.ml.linalg.DenseVector ] )
                wordWeight += ( tf * idf )                
              } else {
                println("Unknown word2vec word " + word)
                // unknownWords += word
              }
          }
        )
        println(" test " +  wordWeight)
        if ( wordWeight != 0.0 ) {
          sentenceVector = sentenceVector / wordWeight
        } else {
          emptyRow = emptyRow + 1          
          println("new Empty Row " + emptyRow + " " + sentenceVector)
        }

        
        contentExtractor.denseBreezeToSpark( sentenceVector ).asInstanceOf[ org.apache.spark.ml.linalg.Vector ]

      }
    )(org.apache.spark.sql.catalyst.encoders.ExpressionEncoder(): org.apache.spark.sql.Encoder[ org.apache.spark.ml.linalg.Vector ])
    // unknownWords.foreach( word => println( word ))
    sentence2vec

  }
  
  def computeKMeans(
      sentence2vec: org.apache.spark.sql.Dataset[org.apache.spark.ml.linalg.Vector], 
      nbClusters:Int,
      sc:org.apache.spark.SparkContext,
      spark: org.apache.spark.sql.SparkSession ) = {
     
    // val word2vec = new org.apache.spark.ml.feature.Word2Vec().setInputCol("filtered")
     
    
    // val word2vecModel = word2vec.fit( sentence )
    // val wordEmbeddings2 = word2vecModel.getVectors.collect().map( row => ( row.getString(0), row.getAs[org.apache.spark.ml.linalg.DenseVector](1))).toMap
    
    
    

        
    var nbIterations = 1000
    var initSteps = 2
    var seed = new org.apache.spark.ml.clustering.KMeans().getSeed
    var scaling = false
    var inputCol = "value"
    var prop = System.getProperty("rcp216.word2vec.nbIterations")
    if ( prop != null ) {
      nbIterations = prop.toInt
    }
    prop = System.getProperty("rcp216.word2vec.initSteps")
    if ( prop != null ) {
      initSteps = prop.toInt
    }
    prop = System.getProperty("rcp216.word2vec.seed")
    if ( prop != null ) {
      seed = prop.toLong
    }
    prop = System.getProperty("rcp216.word2vec.scaling")
    if ( "yes".equals( prop) ) {
      scaling = true
    }
    if ( scaling ) {
      inputCol = "scaled_value"
    }
    val kmeans = new org.apache.spark.ml.clustering.KMeans()
      .setK( nbClusters )
      .setMaxIter( nbIterations)
      .setInitSteps( initSteps )
      .setFeaturesCol( inputCol )
      .setSeed( seed )
      .setInitMode("k-means||")
      .setPredictionCol("topic")
      
    if ( scaling ) {
        val scaler = new StandardScaler()
          .setWithMean( true )
          .setWithStd( true )
          .setInputCol("value")
          .setOutputCol( inputCol )
          .fit( sentence2vec ) 
        val sentenceData = scaler.transform( sentence2vec )
        val kmeansModel = kmeans.fit( sentenceData )
        val kmeanParagraphs = kmeansModel.transform( sentenceData )
        ( kmeanParagraphs, kmeansModel)
    } else {
        val kmeansModel = kmeans.fit( sentence2vec )
        val kmeanParagraphs = kmeansModel.transform( sentence2vec )
        ( kmeanParagraphs, kmeansModel)
    } 
    
      
 //     val l = findSynonyms(kmeansModel, top,  vocab, bWordEmbeddings )

    //saveWSSE( "word2vec-wsse.csv", wsse.toArray)
      
  }
  
  def isEuclidianDistance():Boolean = {
    val distance = System.getProperty("rcp216.word2vec.distance")
    var isEuclidian = false
    if ("euclidian".equals( distance )) {
      isEuclidian = false
    }
    return isEuclidian
  }
  
  
  def findTopicWords( kmeansModel: org.apache.spark.ml.clustering.KMeansModel,
      top:Int, vocab: Array[ String ],
      bWordEmbeddings: org.apache.spark.broadcast.Broadcast[scala.collection.Map[String, org.apache.spark.ml.linalg.Vector]]) = {
    val isEuclidian = isEuclidianDistance()
    val realVocab = vocab.zipWithIndex.filter{ case ( word, index)  => bWordEmbeddings.value.get( word ) != None}
    val topicWords = kmeansModel.clusterCenters.zipWithIndex.map{  case ( cluster, index ) =>
      {
        println("")
        val bCluster = breeze.linalg.DenseVector( cluster.toArray )
        val bClusterLength = breeze.linalg.norm( bCluster )
        // val ordering = Ordering.by[(String, Double( data => data._2)
        val words = realVocab.map{ case (word, index) =>
          {
            val vectr =  bWordEmbeddings.value( word )
            val bVectr = new breeze.linalg.DenseVector[ Double ] (bWordEmbeddings.value( word ).toArray)
            var dist = 0.0
            if ( isEuclidian ) {
              dist =  org.apache.spark.ml.linalg.Vectors.sqdist( cluster, vectr)
            } else {
              dist = ( bCluster dot bVectr ) / ( bClusterLength * breeze.linalg.norm( bVectr ))
            }
            ( index, word, dist )  
          }
        }
        var synonyms:scala.collection.mutable.WrappedArray[(Int, String, Double)] = null
        if ( isEuclidian ) {
          synonyms = words.sortWith( _._3 < _._3).take(top)
        } else {
          synonyms = words.sortWith( _._3 > _._3).take(top)
        } 
        /*
         .foreach(synonym => print(
           " %s (%5.3f),"
           .format(
               synonym._1,
               synonym._2)
          )      
        )
        */
        ( index, synonyms)
      }
    }
    topicWords
    
  }
  
 
  
  def saveMeasures( path: String, measures : Array[ (Int, Double, Double, Double) ]) = {
    val ps = new java.io.PrintStream(new java.io.FileOutputStream(path))
    ps.println("topic\twsse\tword2vect\tUMass")
    measures.foreach( cost => ps.println(cost._1 + "\t"+ + cost._2 +"\t" + cost._3 + "\t" + cost._4 ))
    ps.close()
  }
}