

package com.rcp216.racineTopic

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._

import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.linalg.{Vector, Vectors, SingularValueDecomposition,  Matrix}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.rdd.RDD


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
    
    val mat = new org.apache.spark.mllib.linalg.distributed.RowMatrix(termDocMatrix)
    val svd = mat.computeSVD(maxCluster, computeU=true)
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
    println("vocabulary size: " + vocabulary.length)
    val (termDocMatrix, termIds, docIds, idfs) = termDocumentMatrix( doc, stopWords, vocabulary.length, sc)
    
    val mat = new org.apache.spark.mllib.linalg.distributed.RowMatrix(termDocMatrix)
    val svd = mat.computeSVD(nbClusters, computeU=true)
    saveEigenvalues( "lsa-eigenValues.csv", svd)
    
    val topicWords = findTopicWords(svd, nbClusters, 10, termIds)
    saveTopicWords( "lsa-topicWords-tst.csv", topicWords)
    findTopicMap( svd, 10, termDocMatrix)
    // val topConceptDocs = topDocsInTopConcepts(svd, nbClusters, 10, docIds)
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
      val docTotalTerms = termFreqs.values.sum
      val termScores = termFreqs.filter {
        case (term: String, freq) => ( bTermIds.contains(term) )
      }.map{
        case (term: String, freq) => (bTermIds(term), bIdfs(term) * termFreqs(term) / docTotalTerms)
      }.toSeq
      org.apache.spark.mllib.linalg.Vectors.sparse(bTermIds.size, termScores)
    })
    (vecs, termIds.map(_.swap), docIds, idfs)
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
    val ordering = Ordering.by[(String, Int), Int](_._2)
    docFreqs.top(numTerms)(ordering)
  }

  def trimLeastFrequent(freqs: scala.collection.Map[String, Int], numToKeep: Int): Map[String, Int] = {
    freqs.toArray.sortBy(_._2).take(math.min(numToKeep, freqs.size)).toMap
  }

  def inverseDocumentFrequencies(docFreqs: Array[(String, Int)], numDocs: Int)
    : Map[String, Double] = {
    docFreqs.map{ case (term, count) => (term, math.log(numDocs.toDouble / count))}.toMap
  }

  def saveDocFreqs(path: String, docFreqs: Array[(String, Int)]) {
    val ps = new java.io.PrintStream(new java.io.FileOutputStream(path))
    for ((doc, freq) <- docFreqs) {
      ps.println(s"$doc\t$freq")
    }
    ps.close()
  }
  
  def saveEigenvalues( path: String, svd: org.apache.spark.mllib.linalg.SingularValueDecomposition[org.apache.spark.mllib.linalg.distributed.RowMatrix, org.apache.spark.mllib.linalg.Matrix]) = {
    val ps = new java.io.PrintStream(new java.io.FileOutputStream(path))
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