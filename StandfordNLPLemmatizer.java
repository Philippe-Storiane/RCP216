package com.cnam.rcp216.racineTopics;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;

import edu.stanford.nlp.coref.data.CorefChain;
import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ie.util.*;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.semgraph.*;
import edu.stanford.nlp.trees.*;
import edu.stanford.nlp.util.CoreMap;

import java.util.*;



public class StandfordNLPLemmatizer {

	static String text = "On nous faisait, Arbate, un fidèle rapport:";

	
	public static HashSet<String> initStopPos() {
		HashSet<String> stopPos = new HashSet<String>();
		stopPos.add( "det");
		stopPos.add( "pro");
		stopPos.add( "prel");
		stopPos.add( "puncts");
		stopPos.add( "cla");
		stopPos.add( "clr");
		stopPos.add("prel");
		stopPos.add("prep");
		stopPos.add("csu");
		stopPos.add("adv");
		stopPos.add("coo");
		//
		//stopPos.add("adj");
		stopPos.add("pri");
		return stopPos;
	}
	
	public static HashSet<String> initSentenceEnd() {
		HashSet<String> sentenceEnd = new HashSet<String>();
		sentenceEnd.add( ".");
		sentenceEnd.add("...");
		sentenceEnd.add("?");
		sentenceEnd.add("!");
		return sentenceEnd;		
	}
	
	
	public static HashSet<String> loadWordEmbeddings(String filePath) {
		HashSet<String> words = new HashSet<String>();
		try {
            File f = new File(filePath);
            BufferedReader b = new BufferedReader(new FileReader(f));
            String readLine = "";

            System.out.println("Reading file using Buffered Reader");

            while ((readLine = b.readLine()) != null) {
                String[] tokens = readLine.split("\t");
                words.add( tokens[ 0 ]);
            }
            b.close();
        } catch (IOException e) {
            e.printStackTrace();
        };
        return words;
	}
	
	
	public static void main(String[] args) {
		String folderPath = "C:/Users/a179415/OneDrive - Alliance/Personal/CNAM/RCP 216/";
		String lemmatizedTextFilePath = folderPath + "standford-nlp.txt";
		String racinePlayFilePath = folderPath + "test-nlp-scenic.txt";
		String lemmaFilePath = folderPath + "lefff-3.4.mlex";
		String stopPosFilePath = folderPath + "stanford-stop-pos.csv";
		
		StandfordNLPLemmatizer lemmatizer = new StandfordNLPLemmatizer();
		
		try {
			
			
			
			String readLine;
			
			// Extract lemma definitions
			File lemmaFile = new File(lemmaFilePath);
            BufferedReader lemmaReader = new BufferedReader(new FileReader(lemmaFile));
    		class PosKey {
    			public String text;
    			public String pos;
    			
    			public PosKey( String text, String pos) {
    				this.text = text;
    				this.pos = pos;
    			}

				@Override
				public boolean equals(Object obj) {
					// TODO Auto-generated method stub
					if ( obj  instanceof PosKey) {
						PosKey posKey = ( PosKey) obj;
						return this.text.equals( posKey.text) && this.pos.equals( posKey.pos);
					} else {
						return false;
					}
				}

				@Override
				public int hashCode() {
					// TODO Auto-generated method stub
					int result = 0;
					result = 31 * result + (text != null ? text.hashCode() : 0);
					result = 31 * result + (pos != null ? text.hashCode() : 0);
					return result;
				}
				
				

    			    			    			    		
    		}
    		
    		Map<PosKey,String> posLemmas = new HashMap<PosKey, String>();
    		Map<String, String> defaultLemmas = new HashMap<String, String>();
    		while ((readLine = lemmaReader.readLine()) != null) {
            	String[] lemmaDef = readLine.split("\t");
            	String lemmaText = lemmaDef[ 0 ];
            	String lemmaPos = lemmaDef[ 1];
            	String lemmaValue = lemmaDef[ 2 ];
            	PosKey posKey = new PosKey( lemmaText, lemmaPos);
            	posLemmas.put( posKey, lemmaValue);
            	if ( ! defaultLemmas.containsKey( lemmaText )) {
            		defaultLemmas.put( lemmaText, lemmaValue);
            	}            	
            }
    		
    		PosKey posKey1 = new PosKey("brocardasse","v");
    		PosKey posKey2 = new PosKey("brocardasse","v");
    		System.out.println( "PosKey equality " + ( posKey1 == posKey2));
    		System.out.println( "Lemma found " + posLemmas.get( posKey1 ));
		    Properties props = new Properties();
		    // set the list of annotators to run
		    //props.setProperty("props_fr", "StanfordCoreNLP-french.properties"); 
		    //props.setProperty("tokenize.language","French"); 
		    props.setProperty("tokenize.options", "untokenizable=noneDelete");
		    props.setProperty("tokenize.keepeol","false");// True = garde les sauts de ligne 
		    props.setProperty("tokenize.verbose","false"); // True = affiche les tokens 
		    props.setProperty("ssplit.newlineIsSentenceBreak", "always");
		    //
		    // props.setProperty("annotators", "tokenize,ssplit,pos,lemma");
		    props.setProperty("annotators", "tokenize,ssplit,pos");
		    props.setProperty("tokenize.language", "fr");
		    props.setProperty("pos.model", "edu/stanford/nlp/models/pos-tagger/french/french.tagger");
		    //props.setProperty("parse.model", "edu/stanford/nlp/models/lexparser/frenchFactored.ser.gz");
		    // set a property for an annotator, in this case the coref annotator is being set to use the neural algorithm
		    props.setProperty("coref.algorithm", "neural");
		    // build pipeline
		    StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
		    // create a document object
		    
		    // annnotate the document
		    
			File racinePlayFile = new File(racinePlayFilePath);
            BufferedReader racinePlayReader = new BufferedReader(new FileReader(racinePlayFile));
            FileWriter lemmatizedTextFile = new FileWriter( lemmatizedTextFilePath );
            PrintWriter lemmatizedTextWriter = new PrintWriter( lemmatizedTextFile );
            FileWriter stopPosFile = new FileWriter( stopPosFilePath );
            PrintWriter stopPosWriter = new PrintWriter( stopPosFile );
            stopPosWriter.println("Pos,Text");
        	Set<String> unknownLemmas = new HashSet<String>();
        	HashMap<String, String> stanfordTags = initStanfordTags();
        	HashSet<String> stopPos = initStopPos();
        	HashSet<String> sentenceEnd = initSentenceEnd();
          // HashSet<String> posTags = new HashSet<String>();
        	int wordIndex = 0;
        	int lemmaError = 0;
        	int noLemmaError = 0;
            while ((readLine = racinePlayReader.readLine()) != null) {
            	Annotation annotation = new Annotation( readLine );
            	CoreDocument document = new CoreDocument(annotation);
            	pipeline.annotate(document);

            	List<CoreSentence> sentences = document.sentences();
            	//System.out.println( "number sentences " + sentences.size());
            	for( CoreSentence sentence: document.sentences()) {
            		List<String> tags = sentence.posTags();
            		List<CoreLabel> tokens = sentence.tokens();
            		int length = tokens.size();
            		for(int index =0; index < length; index++) {
            			wordIndex++;
            			String rawText = tokens.get( index ).toString().split("-")[0];
            			String sample = rawText.toLowerCase();
            			String token = stanfordTags.get( tags.get( index ));
            			if ( stopPos.contains( token ) ) {
            				stopPosWriter.println(token + "," + sample);
            				if ( token.equals("puncts") && sentenceEnd.contains(sample )) {
            					lemmatizedTextWriter.println("");
            				}
            				continue;
            			}
            			if ( rawText.toUpperCase().equals(rawText )) {
            				if ( ( ! rawText.matches("^[0-9]+$")) && (rawText.length() > 2)) {
            					lemmatizedTextWriter.print( " " + rawText);
            				}
            				continue;
            			}
                		if ( ( ! sample.matches("^[0-9]+$")) && (sample.length() > 2) ) {
                			
                			PosKey posKey = new PosKey( sample, token);
                			if ( posLemmas.containsKey( posKey )) {
                				lemmatizedTextWriter.print(" " + posLemmas.get( posKey ));
                			} else {
                				lemmaError++;
                				if ( defaultLemmas.containsKey( sample )) {                					
                					lemmatizedTextWriter.print(" " + defaultLemmas.get( sample ));
                				} else {
                					noLemmaError++;
                					unknownLemmas.add( sample );
                					lemmatizedTextWriter.print( " " + rawText);
                				}
                			}
                		}
            		}
            	}
            	lemmatizedTextWriter.println("");
            }
            racinePlayReader.close();
            stopPosWriter.close();
            lemmatizedTextWriter.close();

            for( String word: unknownLemmas ) {
            	System.out.println( word);
            }
            System.out.println("Number of error due to unknown lemma + pos combination: " + lemmaError);
            System.out.println("Number of error due to unknown lemma: " + noLemmaError);
            System.out.println("Number of scanned words: " + wordIndex);
            System.out.println("Number of unknown lemmas: " + unknownLemmas.size());
			
		} catch ( Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}


	private static HashMap<String, String> initStanfordTags() {
		// TODO Auto-generated method stub
		HashMap<String, String> stanfordTags = new HashMap<String,String>();
		stanfordTags.put("NC","nc");
		stanfordTags.put("VPP","v");
        stanfordTags.put("VINF","v");
        stanfordTags.put("N","nc");
        stanfordTags.put("V","v");
        stanfordTags.put("VS","v");
        stanfordTags.put("PRO","pro");
        stanfordTags.put("ADJ","adj");
        stanfordTags.put("VPR","v");
        stanfordTags.put("NPP","nc");
        stanfordTags.put("DETWH","det");
        stanfordTags.put("PROREL","prel"); //TBC
        stanfordTags.put("VIMP","v");
        stanfordTags.put("ADV","adv");
        stanfordTags.put("P","prep");
        stanfordTags.put("DET", "det");
        stanfordTags.put("CS", "csu");
        stanfordTags.put("CLO", "pro");// #TBC vous, nous, les, puisqu'il, lui, leur, 
        stanfordTags.put("I","nc");
        stanfordTags.put("C","pri"); // que, qu', tandis or prel
        stanfordTags.put("CLS","pro");
        stanfordTags.put("PUNC", "puncts");
        stanfordTags.put("CC", "coo");
        stanfordTags.put("ET", "nc");
        stanfordTags.put("ADVWH", "pri");
        stanfordTags.put("ADJWH", "det");
        stanfordTags.put("PROWH","prel"); // TBC
        stanfordTags.put("CL", "cla");
        stanfordTags.put("CLR", "clr");
		return stanfordTags;
	}
}
