

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

class RunLSA extends Serializable {
  

  
  def run( nbClusters:Int, top: Int, numTerms: Int, sc:org.apache.spark.SparkContext, spark: org.apache.spark.sql.SparkSession ) {

    val contentExtractor = new ContentExtractor()
    val paragraphs = contentExtractor.extractContent(sc)
    val doc = contentExtractor.extractRDD( paragraphs, sc )
    val stopWords = contentExtractor.loadStopWords().toSet.asInstanceOf[ Set[String]]
    val (docDF, vocabulary) = contentExtractor.extractDataFrame(paragraphs, sc, spark)
    println("number of paragraphs: " + paragraphs.length)
    println("vocabulary size: " + vocabulary.length)
    val (termDocMatrix, termIds, docIds, idfs) = termDocumentMatrix( doc, stopWords, numTerms, sc)
    
    val mat = new org.apache.spark.mllib.linalg.distributed.RowMatrix(termDocMatrix)
    val svd = mat.computeSVD(nbClusters, computeU=true)
    saveEigenvalues( "lsaEigenValues.csv", svd)
    val topConceptTerms = topTermsInTopConcepts(svd, nbClusters, top, termIds)
    val topConceptDocs = topDocsInTopConcepts(svd, nbClusters, top, docIds)
    for ((terms, docs) <- topConceptTerms.zip(topConceptDocs)) {
      println("Concept terms: " + terms.map(_._1).mkString(", "))
      println("Concept docs: " + docs.map(_._1).mkString(", "))
      println()
    }
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
  
    def topTermsInTopConcepts(svd: org.apache.spark.mllib.linalg.SingularValueDecomposition[org.apache.spark.mllib.linalg.distributed.RowMatrix, org.apache.spark.mllib.linalg.Matrix], numConcepts: Int,
      numTerms: Int, termIds: scala.collection.Map[Int, String]): Seq[Seq[(String, Double)]] = {
    val v = svd.V
    val topTerms = new scala.collection.mutable.ArrayBuffer[Seq[(String, Double)]]()
    val arr = v.toArray
    for (i <- 0 until numConcepts) {
      val offs = i * v.numRows
      val termWeights = arr.slice(offs, offs + v.numRows).zipWithIndex
      val sorted = termWeights.sortBy(-_._1)
      topTerms += sorted.take(numTerms).map{case (score, id) => (termIds(id), score)}
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

  
}