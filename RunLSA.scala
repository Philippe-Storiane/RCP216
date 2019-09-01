

package com.cnam.rcp216.racineTopic

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._

import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.linalg.{Vector, Vectors, SingularValueDecomposition,  Matrix}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.rdd.RDD

import breeze.linalg.{DenseMatrix => BDenseMatrix, DenseVector => BDenseVector,
SparseVector => BSparseVector}


import org.apache.spark.ml.feature.StopWordsRemover

import scala.collection.JavaConversions._
import scala.collection.Map
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashMap

import java.io.{FileOutputStream, PrintStream}

class RunLSA extends AbstractRun {
  
 
  def searchClusterSize(minCluster: Int, maxCluster:Int, sc:org.apache.spark.SparkContext, spark: org.apache.spark.sql.SparkSession) = {
    val contentExtractor = new ContentExtractor()
    val paragraphs = contentExtractor.extractContent(sc)
    val doc = contentExtractor.extractRDD( paragraphs, sc )
    val stopWords = contentExtractor.loadStopWords( sc ).toSet.asInstanceOf[ Set[String]]
    val (docDF, vocabulary) = contentExtractor.extractDataFrame(paragraphs, sc, spark)
    println("number of paragraphs: " + paragraphs.length)
    println("vocabulary size: " + vocabulary.length)
    val (termDocMatrix, termIds, docIds, idfs) = termDocumentMatrix( doc, stopWords, vocabulary.length, sc)
    
    val bWordEmbeddings = CoherenceMeasure.loadWordEmbeddings(sc)
    val corpusPMI = CoherenceMeasure.preprocessUMass( docDF, vocabulary.length )
    var k = 200
    val mat = new org.apache.spark.mllib.linalg.distributed.RowMatrix(termDocMatrix)
    val svd = mat.computeSVD(k , computeU=true)
    val measures = for ( nbCluster <- minCluster to maxCluster ) yield {
      val topicWords = findTopicWords(svd, nbCluster, 10, termIds)
      val topicsDump = topicWords.map{
        case ( topicIndex, words) => {
          val wordIndex = words.map( _._1).toArray
          ( topicIndex, wordIndex)
        }
      }
      val topicsWord2vec = CoherenceMeasure.word2vec(topicsDump, vocabulary, bWordEmbeddings)
      val word2vec = topicsWord2vec.map( _._2).sum / nbCluster
      val uMass = CoherenceMeasure.uMass(topicsDump, corpusPMI).map(_._2).sum / nbCluster

      (nbCluster, word2vec, uMass)
    }
    saveMeasures( "lsa-measures-tst.csv", measures.toArray)
    saveEigenvalues( "lsa-eigenValues.csv", svd)
    
  }
  
  def analyzeCluster( nbClusters:Int, sc:org.apache.spark.SparkContext, spark: org.apache.spark.sql.SparkSession) = {
    val contentExtractor = new ContentExtractor()
    val paragraphs = contentExtractor.extractContent(sc)
    val doc = contentExtractor.extractRDD( paragraphs, sc )
    val stopWords = contentExtractor.loadStopWords( sc ).toSet.asInstanceOf[ Set[String]]
    val (docDF, vocabulary) = contentExtractor.extractDataFrame(paragraphs, sc, spark)
    println("number of paragraphs: " + paragraphs.length)
    println("number of non empty paragraphs: " + doc.count)
    println("vocabulary size: " + vocabulary.length)
    val (termDocMatrix, termIds, docIds, idfs) = termDocumentMatrix( doc, stopWords, vocabulary.length, sc)
    
    val normalizer = new org.apache.spark.mllib.feature.Normalizer
    val normalizedTermDocMatrix = normalizer.transform( termDocMatrix)

    
    val rawKMeansModel = computeKMeans( normalizedTermDocMatrix, nbClusters)
    val rawTopicWords = rawKMeansModel.clusterCenters.zipWithIndex.map{
      case ( clusterCenter, index ) => {
        val arr = clusterCenter.toArray.zipWithIndex.sortBy(-_._1).take(10).map( v => ( v._2, termIds(v._2), v._1))
        val highest:scala.collection.mutable.WrappedArray[(Int, String, Double)] = arr
        ( index, highest)
      }
    }
    saveTopicWords("lsa-raw-topicWords-tst.csv", rawTopicWords)
    var k = 200
    val mat = new org.apache.spark.mllib.linalg.distributed.RowMatrix(termDocMatrix)
    val svd = mat.computeSVD(k , computeU=true)
    saveEigenvalues( "lsa-eigenValues.csv", svd)
    
    val topicWords = findTopicWords(svd, nbClusters, 10, termIds)
    saveTopicWords( "lsa-topicWords-tst.csv", topicWords)
    
    val documentMatrix = multiplyByDiagonalMatrix( svd.U, svd.s)
    val normalizedDocumentMatrix = rowsNormalized( documentMatrix )
    val svdKMeansModel = computeKMeans( normalizedDocumentMatrix.rows, nbClusters)
    
    val wordMatrix = multiplyByDiagonalMatrix( svd.V, svd.s)
    val normalizedWordMatrix = rowsNormalized( wordMatrix )
    val bNormalizedWordMatrix = new BDenseMatrix[Double](
        normalizedWordMatrix.rows,
        normalizedWordMatrix.cols,
        normalizedWordMatrix.toArray)
    val svdTopicWords = svdKMeansModel.clusterCenters.zipWithIndex.map{
      case ( cluster, index) => {
        var bCluster = BDenseVector[Double]( normalizer.transform(cluster).toArray )
        val value = bNormalizedWordMatrix * bCluster
        val weights = value.toArray.zipWithIndex.sortBy(-_._1).take(10).map( v => (v._2, termIds( v._2), v._1))
        val highest:scala.collection.mutable.WrappedArray[(Int, String, Double)] = weights
        ( index, highest)

      }
    }
    saveTopicWords( "lsa-svd-topicWords-tst.csv", svdTopicWords)
    findTopicMap( svd, 10, termDocMatrix)
    // val topConceptDocs = topDocsInTopConcepts(svd, nbClusters, 10, docIds)
  }
  
  def computeKMeans( data:RDD[ Vector ], nbClusters: Int ) = {
    var nbIterations = 1000
    var initSteps = 2
    var seed = new org.apache.spark.ml.clustering.KMeans().getSeed
    var scaling = false
    var inputCol = "value"
    var prop = System.getProperty("rcp216.lsa.nbIterations")
    if ( prop != null ) {
      nbIterations = prop.toInt
    }
    prop = System.getProperty("rcp216.lsa.initSteps")
    if ( prop != null ) {
      initSteps = prop.toInt
    }
    prop = System.getProperty("rcp216.lsa.initSteps")
    if ( prop != null ) {
      initSteps = prop.toInt
    }
    prop = System.getProperty("rcp216.lsa.seed")
    if ( prop != null ) {
      seed = prop.toLong
    }

    val kmeans = new org.apache.spark.mllib.clustering.KMeans()
      .setK( nbClusters )
      .setSeed( seed )
      .setMaxIterations(nbIterations)
      .setInitializationSteps(initSteps)
      .setInitializationMode("k-means||")
   kmeans.run( data )
  }

    /**
   * Selects a row from a matrix.
   */
  def row(mat: BDenseMatrix[Double], index: Int): Seq[Double] = {
    (0 until mat.cols).map(c => mat(index, c))
  }

  /**
   * Selects a row from a matrix.
   */
  def row(mat: Matrix, index: Int): Seq[Double] = {
    val arr = mat.toArray
    (0 until mat.numCols).map(i => arr(index + i * mat.numRows))
  }

  /**
   * Selects a row from a distributed matrix.
   */
  def row(mat: RowMatrix, id: Long): Array[Double] = {
    mat.rows.zipWithUniqueId.map(_.swap).lookup(id).head.toArray
  }

  
  def findTopicWords( svd:org.apache.spark.mllib.linalg.SingularValueDecomposition[org.apache.spark.mllib.linalg.distributed.RowMatrix, org.apache.spark.mllib.linalg.Matrix],
      nbClusters:Int,
      top:Int, termIds:scala.collection.Map[Int, String]) = {
    val topConceptTerms = topTermsInTopConcepts(svd, nbClusters, 10, termIds)
    val topicWords = topConceptTerms.zipWithIndex.toArray.map{
      case ( seq, index ) => {
        val arr = seq.toArray
        val synonyms:scala.collection.mutable.WrappedArray[(Int, String, Double)] = arr
        (index, synonyms)
      }
    }
    topicWords
  }
  
  def findTopicMap( svd:org.apache.spark.mllib.linalg.SingularValueDecomposition[org.apache.spark.mllib.linalg.distributed.RowMatrix, org.apache.spark.mllib.linalg.Matrix],
                    top: Int,
                    termDocMatrix:org.apache.spark.rdd.RDD[org.apache.spark.mllib.linalg.Vector]) = {
    val v = svd.V
    val mat = breeze.linalg.DenseMatrix( v.toArray)
    val indices = for( index <- 0 to v.numCols - 1) yield index
    val topicMap = termDocMatrix.zipWithIndex.map{
      case ( doc, index) => {
        val bDoc = breeze.linalg.SparseVector[Double]( doc.toArray)
        val bDocLength = breeze.linalg.norm( bDoc)
        val weights = indices.map( index => ( index, ( mat(::, index) dot bDoc ) / bDocLength))
        val topWeights = weights.sortBy( _._2).take( top )
        (index.toLong, topWeights.toArray)
      }
    }.collect()
    saveTopicMap("lsa-topicMap-tst.csv", topicMap)
  }
    
  def termDocumentMatrix(docs: org.apache.spark.rdd.RDD[(String, Array[String])], stopWords: Set[String], numTerms: Int,
      sc: org.apache.spark.SparkContext): (org.apache.spark.rdd.RDD[org.apache.spark.mllib.linalg.Vector], Map[Int, String], scala.collection.mutable.Map[Long, String], Map[String, Double]) = {
    val docTermFreqs = docs.mapValues(terms => {
      val termFreqsInDoc = terms.foldLeft(new scala.collection.mutable.HashMap[String, Int]()) {
        (map, term) => map += term -> (map.getOrElse(term, 0) + 1)
      }
      termFreqsInDoc
    })

    docTermFreqs.cache()
    val docIds = docTermFreqs.map(_._1).zipWithUniqueId().map(_.swap).collectAsMap().asInstanceOf[ scala.collection.mutable.Map[Long, String] ]

    val docFreqs = documentFrequenciesDistributed(docTermFreqs.map(_._2), numTerms)
    println("Number of terms: " + docFreqs.size)
    saveDocFreqs("docfreqs.tsv", docFreqs)
    val docFreqsMap = docFreqs.toMap

    val numDocs = docIds.size

    val idfs = inverseDocumentFrequencies(docFreqs, numDocs)

    // Maps terms to their indices in the vector
    val termIds = idfs.keys.zipWithIndex.toMap

    val bIdfs = sc.broadcast(idfs).value
    val bTermIds = sc.broadcast(termIds).value

    val vecs: org.apache.spark.rdd.RDD[org.apache.spark.mllib.linalg.Vector] = docTermFreqs.map(_._2).map(termFreqs => {
      val termFreqsFiltered = termFreqs.filter {
        case (term: String, freq) => ( bTermIds.contains(term) )
      }
      val docTotalTerms = termFreqsFiltered.values.sum
      val termScores = termFreqsFiltered.map{
        case (term: String, freq) => (bTermIds(term), bIdfs(term) * termFreqs(term) / docTotalTerms)
      }.toSeq
      org.apache.spark.mllib.linalg.Vectors.sparse(bTermIds.size, termScores)
    })
    var minNonzeros = 0
    var prop = System.getProperty("rcp216.minNonzeros")
    if ( prop != null ) {
      minNonzeros = prop.toInt
    }
    (vecs.filter( vect => vect.numNonzeros > minNonzeros ), termIds.map(_.swap), docIds, idfs)
  }

  def documentFrequencies(docTermFreqs: org.apache.spark.rdd.RDD[scala.collection.mutable.HashMap[String, Int]]): scala.collection.mutable.HashMap[String, Int] = {
    val zero = new scala.collection.mutable.HashMap[String, Int]()
    def merge(dfs: scala.collection.mutable.HashMap[String, Int], tfs: scala.collection.mutable.HashMap[String, Int])
      : scala.collection.mutable.HashMap[String, Int] = {
      tfs.keySet.foreach { term =>
        dfs += term -> (dfs.getOrElse(term, 0) + 1)
      }
      dfs
    }
    def comb(dfs1: scala.collection.mutable.HashMap[String, Int], dfs2: scala.collection.mutable.HashMap[String, Int])
      : scala.collection.mutable.HashMap[String, Int] = {
      for ((term, count) <- dfs2) {
        dfs1 += term -> (dfs1.getOrElse(term, 0) + count)
      }
      dfs1
    }
    docTermFreqs.aggregate(zero)(merge, comb)
  }

  def documentFrequenciesDistributed(docTermFreqs: org.apache.spark.rdd.RDD[scala.collection.mutable.HashMap[String, Int]], numTerms: Int)
      : Array[(String, Int)] = {
    val docFreqs = docTermFreqs.flatMap(_.keySet).map((_, 1)).reduceByKey(_ + _, 15)
    // val ordering = Ordering.by[(String, Int), Int](_._2)
    // docFreqs.top(numTerms)(ordering)
    var minDF = 2
    val prop = System.getProperty("rcp216.minDF")
    if ( prop != null ) {
      minDF = prop.toInt
    }
    docFreqs.filter( row => row._2 >= minDF ).collect()
  }

  def trimLeastFrequent(freqs: scala.collection.Map[String, Int], numToKeep: Int): Map[String, Int] = {
    freqs.toArray.sortBy(_._2).take(math.min(numToKeep, freqs.size)).toMap
  }

  def inverseDocumentFrequencies(docFreqs: Array[(String, Int)], numDocs: Int)
    : Map[String, Double] = {
    docFreqs.map{ case (term, count) => (term, math.log( ( numDocs.toDouble + 1 ) / ( count + 1 ) ))}.toMap
  }

  
  def multiplyByDiagonalMatrix(mat: Matrix, diag: Vector): BDenseMatrix[Double] = {
    val sArr = diag.toArray
    new BDenseMatrix[Double](mat.numRows, mat.numCols, mat.toArray)
      .mapPairs{case ((r, c), v) => v * sArr(c)}
  }

  
   /**
   * Finds the product of a distributed matrix and a diagonal matrix represented by a vector.
   */
  def multiplyByDiagonalMatrix(mat: RowMatrix, diag: Vector): RowMatrix = {
    val sArr = diag.toArray
    new RowMatrix(mat.rows.map(vec => {
      val vecArr = vec.toArray
      val newArr = (0 until vec.size).toArray.map(i => vecArr(i) * sArr(i))
      Vectors.dense(newArr)
    }))
  }
  
  
   def rowsNormalized(mat: BDenseMatrix[Double]): BDenseMatrix[Double] = {
    val newMat = new BDenseMatrix[Double](mat.rows, mat.cols)
    for (r <- 0 until mat.rows) {
      val length = math.sqrt((0 until mat.cols).map(c => mat(r, c) * mat(r, c)).sum)
      (0 until mat.cols).map(c => newMat.update(r, c, mat(r, c) / length))
    }
    newMat
  }
   
  
    /**
   * Returns a distributed matrix where each row is divided by its length.
   */
  def rowsNormalized(mat: RowMatrix): RowMatrix = {
    new RowMatrix(mat.rows.map(vec => {
      val length = math.sqrt(vec.toArray.map(x => x * x).sum)
      Vectors.dense(vec.toArray.map(_ / length))
    }))
  }

   

  def saveDocFreqs(path: String, docFreqs: Array[(String, Int)]) {
    val ps = new java.io.PrintStream(new java.io.FileOutputStream(path))
    ps.println("word\tfreq")
    for ((doc, freq) <- docFreqs) {
      ps.println(s"$doc\t$freq")
    }
    ps.close()
  }
  
  def saveEigenvalues( path: String, svd: org.apache.spark.mllib.linalg.SingularValueDecomposition[org.apache.spark.mllib.linalg.distributed.RowMatrix, org.apache.spark.mllib.linalg.Matrix]) = {
    val ps = new java.io.PrintStream(new java.io.FileOutputStream(path))
    ps.println("eigenvalue")
    svd.s.toArray.map( value => {
        ps.println( value)
      }
    )
    ps.close()
  }
  
    def topTermsInTopConcepts(svd: org.apache.spark.mllib.linalg.SingularValueDecomposition[org.apache.spark.mllib.linalg.distributed.RowMatrix, org.apache.spark.mllib.linalg.Matrix],
        numConcepts: Int,
        numTerms: Int,
        termIds: scala.collection.Map[Int, String]): Seq[Seq[(Int,String, Double)]] = {
    val v = svd.V
    val topTerms = new scala.collection.mutable.ArrayBuffer[Seq[(Int,String, Double)]]()
    val arr = v.toArray
    for (i <- 0 until numConcepts) {
      val offs = i * v.numRows
      val termWeights = arr.slice(offs, offs + v.numRows).zipWithIndex.map{
        case ( value, index) => ( math.sqrt( value * value), index)
      }
      val sorted = termWeights.sortBy(-_._1)
      topTerms += sorted.take(numTerms).map{case (score, id) => (id,  termIds(id), score)}
    }
    topTerms
  }

  def topDocsInTopConcepts(svd: org.apache.spark.mllib.linalg.SingularValueDecomposition[org.apache.spark.mllib.linalg.distributed.RowMatrix, org.apache.spark.mllib.linalg.Matrix], numConcepts: Int,
      numDocs: Int, docIds: scala.collection.Map[Long, String]): Seq[Seq[(String, Double)]] = {
    val u  = svd.U
    val topDocs = new scala.collection.mutable.ArrayBuffer[Seq[(String, Double)]]()
    for (i <- 0 until numConcepts) {
      val docWeights = u.rows.map(_.toArray(i)).zipWithUniqueId
      topDocs += docWeights.top(numDocs).map{case (score, id) => (docIds(id), score)}
    }
    topDocs
  }

  def saveMeasures( path: String, measures : Array[ (Int, Double, Double) ]) = {
    val ps = new java.io.PrintStream(new java.io.FileOutputStream(path))
    ps.println("topic\tword2vect\tUMass")
    measures.foreach( cost => ps.println(cost._1 + "\t"+ + cost._2 +"\t" + cost._3 ))
    ps.close()
  }
}