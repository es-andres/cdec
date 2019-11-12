package comparer;

import static org.junit.jupiter.api.Assertions.assertNotEquals;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.ProtocolException;
import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;
import java.util.stream.IntStream;

import org.apache.commons.math3.ml.distance.CanberraDistance;
import org.apache.commons.math3.ml.distance.ChebyshevDistance;
import org.apache.commons.math3.ml.distance.DistanceMeasure;
import org.apache.commons.math3.ml.distance.EarthMoversDistance;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.apache.commons.math3.ml.distance.ManhattanDistance;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;

import common.ECBWrapper;
import common.Globals;
import common.Main;
import edu.emory.clir.clearnlp.util.StringUtils;
import edu.stanford.nlp.coref.data.CorefChain;
import edu.stanford.nlp.coref.data.CorefChain.CorefMention;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.pipeline.CoreSentence;
import me.tongfei.progressbar.ProgressBar;
import naf.EventNode;
import naf.NafDoc;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class EvPairDataset {
	
	public final Attribute label = new Attribute("class", new ArrayList<String>(Arrays.asList(new String[] {"0", "1"})));
	static final Logger LOGGER = Logger.getLogger(EvPairDataset.class.getName());
	public Instances data;
	
	private Instances makeInstances(int size) {
		/*
		 * add attributes
		 */
		String[] realAttrs = {"word_dist_sim", "ev_sentence_sim", "head_word_vec"};
		ArrayList<Attribute> cols = new ArrayList<Attribute>();
		for(String key : Arrays.asList(realAttrs))
			cols.add(new Attribute(key));
		String[] boolAttrs = {"NN", "VV", "NV"};
		List bool = Arrays.asList("0","1");
		for(String key : Arrays.asList(boolAttrs))
			cols.add(new Attribute(key, bool));
		cols.add(label);
		
		/*
		 * make dataset
		 */
		this.data = new Instances("ev_pairs", cols, size);
		data.setClass(label);
		
		return data;
	}
	
	public Instances makeDataset(ECBWrapper dataWrapper, List<List<EventNode>> pairs, boolean lemmatize, String[] pos) {
		LOGGER.info("Making dataset");
		Instances data = makeInstances(pairs.size());
		
		for(List<EventNode> pair : pairs) {
			Instance vec = evPairVector(dataWrapper, pair,lemmatize, pos);
			data.add(vec);
		}
		return data;
	}
	
	public Instance evPairVector(ECBWrapper dataWrapper, List<EventNode> pair, boolean lemmatize, String[] pos) {
		Instance instance = new DenseInstance(this.data.numAttributes());
		EventNode ev1 = pair.get(0);
		EventNode ev2 = pair.get(1);
		NafDoc doc1 = dataWrapper.docs.get(ev1.file.getName());
		NafDoc doc2 = dataWrapper.docs.get(ev2.file.getName());
		
		List<IndexedWord> ev1Text = mainEvText(ev1, doc1);
		List<IndexedWord> ev2Text = mainEvText(ev2, doc2);
		List<CoreSentence> ev1Sentences = Arrays.asList(evMainSentence(ev1, doc1));
		List<CoreSentence> ev2Sentences = Arrays.asList(evMainSentence(ev2, doc2));
		
		List<CoreSentence> sentenceCorpus = makeSentCorpus(ev1Sentences, ev2Sentences, ev1.file.equals(ev2.file));
		List<CoreSentence> docCorpus = new LinkedList<CoreSentence>();
		if(ev1.file.equals(ev2.file))
			docCorpus.addAll(doc1.coreDoc.sentences());
		else {
			docCorpus.addAll(doc1.coreDoc.sentences());
			docCorpus.addAll(doc2.coreDoc.sentences());
		}

		/*
		 * event text distribution distance
		 */
		int ngrams = 1;
		TFIDF tfidf = new TFIDF(sentenceCorpus, lemmatize, pos, ngrams);
		DistanceMeasure dist = new EuclideanDistance();
		instance.setValue(this.data.attribute("word_dist_sim"), dist.compute(tfidf.makeEvVector(ev1Text, ev1Sentences),
																			 tfidf.makeEvVector(ev2Text, ev2Sentences)));
		
		String s_id1 = doc1.toks.get(doc1.mentionIdToTokens.get(ev1.m_id).get(0)).get("sentence");
		String s_id2 = doc2.toks.get(doc2.mentionIdToTokens.get(ev2.m_id).get(0)).get("sentence");
		if(ev1.file.equals(ev2.file) && s_id1.equals(s_id2)) {
			instance.setValue(this.data.attribute("ev_sentence_sim"), 0.0);
		}
		else {
			double d =  dist.compute(tfidf.makeSentVector(ev1Sentences, docCorpus),tfidf.makeSentVector(ev2Sentences, docCorpus));
			instance.setValue(this.data.attribute("ev_sentence_sim"), d);
		}
		
		/*
		 * avg. head lemma w2v distance
		 */
		String delim = "MAKE_LIST";
		String request = String.format("%1$s/?f=%2$s&p1=%3$s&p2=%4$s", Globals.W2V_SERVER, 
										"n_similarity", 
										delim + doc1.getHeadText(ev1.m_id), 
										delim + doc2.getHeadText(ev2.m_id));
		instance.setValue(this.data.attribute("head_word_vec"), Double.parseDouble(getHTML(request)));
		
		/*
		 * pos
		 */
		HashSet<String> pos1 = posSet(ev1Text);
		HashSet<String> pos2 = posSet(ev2Text);
		boolean NN = pos1.contains("N") && pos2.contains("N");
		boolean VV = pos1.contains("V") && pos2.contains("V");
		boolean NV = (pos1.contains("N") && pos2.contains("V") || (pos1.contains("V") && pos2.contains("N")));
		instance.setValue(this.data.attribute("NN"), NN ? "1" : "0");
		instance.setValue(this.data.attribute("VV"), VV ? "1" : "0");
		instance.setValue(this.data.attribute("NV"), NV ? "1" : "0");
			
		
		instance.setValue(this.data.attribute("class"), ev1.corefers(ev2) ? "1" : "0");
		
		
		
		return instance;
	}
	
	private HashSet<String> posSet(List<IndexedWord> evText) {
		HashSet<String> pos = new HashSet<String>();
		for(IndexedWord w : evText)
			pos.add(w.tag().substring(0, 1));
		return pos;
	}
	
	/**
	 * trigger and deps of main event
	 * @param ev
	 * @param set
	 * @return
	 */
	public static List<IndexedWord> mainEvText(EventNode ev, NafDoc doc){
		List<IndexedWord> text = new LinkedList<IndexedWord>();

		for(String level : doc.mIdToCoreEvText.get(ev.m_id).keySet()) {
			for(int tok_idx : doc.mIdToCoreEvText.get(ev.m_id).get(level).navigableKeySet())
				text.add(doc.mIdToCoreEvText.get(ev.m_id).get(level).get(tok_idx));
		}
		
		return text;
	}
	
	/**
	 * corefering entities of trigger and deps
	 * @param ev
	 * @param set
	 * @return
	 */
	private List<IndexedWord> corefEvText(EventNode ev, NafDoc doc){
		List<IndexedWord> text = new LinkedList<IndexedWord>();

		for(String level : doc.mIdToEntCorefs.get(ev.m_id).keySet()) { // trigger or deps
			for(CorefChain chain : doc.mIdToEntCorefs.get(ev.m_id).get(level).values()) {
				for(CorefMention ent : chain.getMentionsInTextualOrder()) {
					CoreSentence sent = doc.coreDoc.sentences().get(ent.sentNum - 1);
	    			for(int tok_idx : IntStream.range(ent.startIndex - 1, ent.endIndex - 1).toArray()) {
	    				text.add(new IndexedWord(sent.tokens().get(tok_idx)));
	    			}
				}
			}
		}
		return text;
	}
	
	/**
	 * get sentence event appears in
	 * @param ev
	 * @param set
	 * @return
	 */
	public static CoreSentence evMainSentence(EventNode ev, NafDoc doc){
		int s_id = doc.mIdToCoreEvText.get(ev.m_id).get("trigger").firstEntry().getValue().sentIndex();

		return doc.coreDoc.sentences().get(s_id);
	}
	
	/**
	 * 
	 * @param ev
	 * @param set
	 * @return
	 */
	private List<CoreSentence> evCorefSentences(EventNode ev, NafDoc doc, int mainSentId){
		HashSet<Integer> seenSentences = new HashSet<Integer>();
		List<CoreSentence> sentences = new LinkedList<CoreSentence>();
		
		for(String level : doc.mIdToEntCorefs.get(ev.m_id).keySet()) {
			for(CorefChain chain : doc.mIdToEntCorefs.get(ev.m_id).get(level).values()) {
				for(CorefMention ent : chain.getMentionsInTextualOrder()) {
					int s_id = ent.sentNum - 1;
					if(s_id == mainSentId)
						continue;
					if(seenSentences.add(s_id))
						sentences.add(doc.coreDoc.sentences().get(s_id));
				}
			}
		}
		return sentences;
	}
	
	public static List<CoreSentence> makeSentCorpus(List<CoreSentence> c1, List<CoreSentence> c2, boolean sameDoc){
		List<CoreSentence> corpus = new LinkedList<CoreSentence>();
		if(!sameDoc) { // different documents, use all sentences
			corpus.addAll(c1);
			corpus.addAll(c2);
		}
		else { // same document, remove duplicate sentences
			HashSet<Integer> seenSentences = new HashSet<Integer>();
			for(CoreSentence s : c1) {
				if(seenSentences.add(s.tokens().get(0).sentIndex()))
					corpus.add(s);
			}
			for(CoreSentence s : c2) {
				if(seenSentences.add(s.tokens().get(0).sentIndex()))
					corpus.add(s);
			}
		}
			
		return corpus;
	}
	
	public static String getHTML(String urlToRead) {
		StringBuilder result = new StringBuilder();
		URL url = null;
		HttpURLConnection conn = null;
		BufferedReader rd = null;
		String line;

		try {
			url = new URL(urlToRead);
			conn = (HttpURLConnection) url.openConnection();
			conn.setRequestMethod("GET");
			rd = new BufferedReader(new InputStreamReader(conn.getInputStream()));
			while ((line = rd.readLine()) != null) {
				result.append(line);
			}
			rd.close();
		} 
		catch (IOException e) {
			e.printStackTrace();
		}
		
		return result.toString();
	}


}
