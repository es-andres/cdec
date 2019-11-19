package feature_extraction;

import static org.junit.jupiter.api.Assertions.assertNotEquals;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;

import common.Globals;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.pipeline.CoreSentence;


public class EventFeatures {
	
	public HashMap<String, Integer> dict;
	public final boolean lemmatize;
	public final HashSet<String> pos;
	public final int ngrams;
	public List<List<String>> corpus;
	public HashMap<String, Integer> freqs;
	public double numGrams;

	public EventFeatures(List<CoreSentence> docs, boolean lemmatize, String[] pos, int ngrams) {
		this.lemmatize = lemmatize;
		this.pos = new HashSet<String>(Arrays.asList(pos));
		this.ngrams = ngrams;
		this.corpus = new ArrayList<List<String>>();
		this.dict = this.makeDict(docs); // this.corpus is populated in "filter" function, and contains ngrams
	}

	
	private HashMap<String, Integer> makeDict(List<CoreSentence> corpus) {
		HashMap<String, Integer> dict = new HashMap<String, Integer>();
		HashSet<String> vocab = this.filterAndNgram(corpus);
		int idx = 0;
		for(String word : vocab)
			dict.put(word, idx++);

		return dict;
	}
	
	private HashSet<String> filterAndNgram(List<CoreSentence> corpus){
		ArrayList<ArrayList<String>> cleanCorpus = new ArrayList<ArrayList<String>>();
		for(CoreSentence doc : corpus)
			cleanCorpus.add(clean(doc));
		HashSet<String> vocab = new HashSet<String>();
		
		for(ArrayList<String> doc : cleanCorpus) {
			int n_copy = this.ngrams;
			// unigrams
			vocab.addAll(doc);
			this.corpus.add(doc);
			
			// ngrams
			while(n_copy > 1) {
				ArrayList<String> grams = ngrams(doc, n_copy--);
				vocab.addAll(grams);
				doc.addAll(grams);
			}
		}
		return vocab;
	}

	
	public ArrayList<String> ngrams(CoreSentence s, int n, boolean lemmatize, HashSet<String> pos) {
		ArrayList<String> words = new ArrayList<String>();
		for(CoreLabel tok : s.tokens()) {
			String word = cleanTok(tok, lemmatize, pos);
			if(word != null)
				words.add(word);
		}
		return ngrams(words, n);
	}
	
	public static String cleanTok(CoreLabel tok, boolean lemmatize, HashSet<String> pos) {

		String clean = "";
		if(pos.isEmpty()) {
			if(lemmatize)
				clean = tok.lemma();
			else
				clean = tok.originalText();
		}
		else if(pos.contains(tok.tag().subSequence(0, 1))) {
			if(lemmatize)
				clean = tok.lemma();
			else
				clean = tok.originalText();
		}
		
		clean = clean.replaceAll("[^a-zA-Z0-9]", "");
		if(clean.length() == 0 || Globals.strIgnore.contains(clean) )
			return null;
		return clean.toLowerCase();
	}

	
	/**
	 * @param doc list of strings
	 * @param term String represents a term
	 * @return term frequency of term in document
	 */
	public double tf(String term, List<String> doc) {
		double result = 0;
		for (String word : doc) {
			if (term.equals(word))
				result++;
		}
		assertNotEquals(result, 0);
		return result / doc.size();
	}

	
	/**
	 * @param docs list of list of strings represents the dataset
	 * @param term String represents a term
	 * @return the inverse term frequency of term in documents
	 */
	public double idf(String term, List<List<String>> docs) {
		double n = 0;
		for (List<String> doc : docs) {
			for (String word : doc) {
				if (term.equals(word)) {
					n++;
					break;
				}
			}
		}
		double idf = Math.log(docs.size() / n);
		assertNotEquals(n, 0.0);
		return idf;
	}
	
	private ArrayList<String> clean(CoreSentence doc){
		ArrayList<String> cleanSent = new ArrayList<String>();
		for(CoreLabel tok : doc.tokens()) {
			String word = cleanTok(tok, this.lemmatize, this.pos);
			if(word != null)
				cleanSent.add(word);
		}
		
		return cleanSent;
	}
	
	public double[] makeSentVector(List<CoreSentence> sent, List<CoreSentence> docs) {
		List<String> words = new LinkedList<String>();
		for(CoreSentence s : sent) {
			words.addAll(clean(s));
		}
		List<List<String>> strDocs = new ArrayList<List<String>>();
		HashMap<String, Integer> vocabDict = new HashMap<String, Integer>();
		int i = 0;
		for(CoreSentence s : docs) {
			ArrayList<String> strSent = clean(s);
			for(String w : strSent) {
				if(!vocabDict.containsKey(w))
					vocabDict.put(w, i++);
			}
			strDocs.add(strSent);
		}
		double[] vec = new double[vocabDict.size()];
		for(String w : words) {
			vec[vocabDict.get(w)] = tf(w, words)* idf(w, strDocs);
		}

		return vec;
	}
	
	public double[] makeEvVector2(List<IndexedWord> ev, List<CoreSentence> sents) {
		List<String> words = new LinkedList<String>();
		for(IndexedWord t : ev) {
			String word = cleanTok(new CoreLabel(t), this.lemmatize, this.pos);
			if(word != null)
				words.add(word);
		}
		List<List<String>> strDocs = new ArrayList<List<String>>();
		HashMap<String, Integer> vocabDict = new HashMap<String, Integer>();
		int i = 0;
		for(CoreSentence s : sents) {
			ArrayList<String> strSent = clean(s);
			for(String w : strSent) {
				if(!vocabDict.containsKey(w))
					vocabDict.put(w, i++);
			}
			strDocs.add(strSent);
		}
		double[] vec = new double[vocabDict.size()];
		for(String w : words) {
			vec[vocabDict.get(w)] = tf(w, words)* idf(w, strDocs);
		}
		return vec;
	}
	public double[] makeEvVector(List<IndexedWord> evText, List<CoreSentence> doc) {
		/*
		 * document
		 */
		List<String> cleanDoc = new ArrayList<String>();
		for(CoreSentence s : doc) {
			// unigrams
			ArrayList<String> cleanSent = clean(s);
			cleanDoc.addAll(clean(s));
			
			// ngrams
			int n_copy = this.ngrams;
			while(n_copy > 1)
				cleanDoc.addAll(ngrams(cleanSent, n_copy--));
		}
		HashSet<String> docVocab = new HashSet<String>(cleanDoc);
		/*
		 * event
		 */
		HashMap<Integer, ArrayList<String>> sIdToEvText = new HashMap<Integer, ArrayList<String>>(); 
		List<String> evNGrams = new ArrayList<String>();
		// partition by sentence for ngrams
		for(IndexedWord tok : evText) {
			if(!sIdToEvText.containsKey(tok.sentIndex()))
				sIdToEvText.put(tok.sentIndex(), new ArrayList<String>());
			String word = cleanTok(new CoreLabel(tok), this.lemmatize, this.pos);
			if(word != null)
				sIdToEvText.get(tok.sentIndex()).add(word);
		}
		for(int s_id : sIdToEvText.keySet()) {
			evNGrams.addAll(sIdToEvText.get(s_id));
			int n_copy = this.ngrams;
			while(n_copy > 1) {
				ArrayList<String> grams = ngrams(sIdToEvText.get(s_id), n_copy--);
				for(String g : grams) {
					if(docVocab.contains(g)) // evText might have non-contiguous ngrams
						evNGrams.add(g);
				}
			}
		}
		double[] vector = new double[this.dict.size()];
		for(String gram : evNGrams) {
			vector[this.dict.get(gram)] += 1.0/evNGrams.size();
		}

		return vector;
	}
	
	public static ArrayList<String> ngrams(ArrayList<String> words, int n) {
		if(n <= 1) {
			return new ArrayList<String>(words);
		}
		
		ArrayList<String> ngrams = new ArrayList<String>();
		
		int c = words.size();
		for(int i = 0; i < c; i++) {
			if((i + n - 1) < c) {
				int stop = i + n;
				String ngramWords = words.get(i);
				
				for(int j = i + 1; j < stop; j++) {
					ngramWords +=" "+ words.get(j);
				}
				
				ngrams.add(ngramWords);
			}
		}
		
		return ngrams;
	}



}
