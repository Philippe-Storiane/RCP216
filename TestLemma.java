package com.cnam.rcp216.racineTopics;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashSet;
import java.util.List;

import opennlp.tools.postag.POSModel;
import opennlp.tools.postag.POSTagger;
import opennlp.tools.postag.POSTaggerME;
import opennlp.tools.tokenize.Tokenizer;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.util.InvalidFormatException;
import opennlp.tools.util.Sequence;
import opennlp.tools.chunker.Chunker;
import opennlp.tools.chunker.ChunkerME;
import opennlp.tools.chunker.ChunkerModel;


public class TestLemma {

	static String text = "On nous faisait, Arbate, un fidèle rapport:";
	
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
		String wordEmbeddingsFilePath = folderPath + "frWac_non_lem_no_postag_no_phrase_200_skip_cut100.txt";
		String racinePlayFilePath = folderPath + "test-nlp.txt";
		TestLemma test = new TestLemma();
		//InputStream tokenInputFile = test.getClass().getResourceAsStream("/opennlp/models/fr/fr-token.bin");
		InputStream tokenInputFile = test.getClass().getResourceAsStream("/opennlp/models/fr/frenchTreebank-cmmndLn-fr-token.bin");
		TokenizerModel tokenizerModel;
		InputStream posInputFile = test.getClass().getResourceAsStream("/opennlp/models/fr/fr-pos-maxent.bin");
		POSModel posModel;
		POSTagger posTagger;
		InputStream lemmaInputFile = test.getClass().getResourceAsStream("/opennlp/models/fr/fr-lemma.bin");
		InputStream chunkerInputFile = test.getClass().getResourceAsStream("/opennlp/models/fr/fr-chunk.bin");
		ChunkerModel chunkerModel;
		Chunker chunker;
		String filePath="";
		
		try {
			// tokenizerModel = new TokenizerModel( tokenInputFile );
			tokenizerModel = new TokenizerModel( tokenInputFile );
			Tokenizer tokenizer = new TokenizerME( tokenizerModel );
			posModel = new POSModel( posInputFile );
			posTagger = new POSTaggerME( posModel );
			chunkerModel = new ChunkerModel( chunkerInputFile );
			chunker = new ChunkerME( chunkerModel );
			File f = new File(racinePlayFilePath);
            BufferedReader b = new BufferedReader(new FileReader(f));
            String readLine;
            HashSet<String> unknwownWords = new HashSet<String>();
            HashSet<String> wordEmbeddings = loadWordEmbeddings( wordEmbeddingsFilePath );
            while ((readLine = b.readLine()) != null) {
				String[] tokens = tokenizer.tokenize( readLine );
				/*
				String[] tags = posTagger.tag( tokens);
				System.out.println("Tokenize test");
				for( String word: tokens ) {
					System.out.println(word);
				}
				System.out.println("\n\nPartOfSpeach test");
				for( String tag: tags) {
					System.out.println( tag );
				}
				System.out.println("\n\nChunk test");
				String[] chunks = chunker.chunk( tokens,  tags);
				tokenInputFile.close();
				for( String chunk: chunks) {
					System.out.println( chunk );
				}
				*/
				for ( String token: tokens) {
					String sample = token.toLowerCase();
					if ( ! wordEmbeddings.contains( sample ) && ( ! sample.matches("^[0-9]+$"))) {
						unknwownWords.add( sample);
					}
				}
            }
            b.close();
            for( String word: unknwownWords ) {
            	System.out.println( word);
            }
            System.out.println("Number of unknown words: " + unknwownWords.size());
		} catch ( Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
}
