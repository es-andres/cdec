package feature_extraction;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

import java.net.HttpURLConnection;
import java.net.URL;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.logging.Logger;

import org.apache.commons.math3.ml.distance.DistanceMeasure;
import org.apache.commons.math3.ml.distance.EuclideanDistance;

import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

import common.Globals;
import ecb_utils.ECBDoc;
import ecb_utils.ECBWrapper;
import ecb_utils.EventNode;

import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.pipeline.CoreSentence;

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
		String[] realAttrs = {"word_distribution_sim", "ev_sentence_sim", "head_word_vec_sim"};
		ArrayList<Attribute> cols = new ArrayList<Attribute>();
		for(String key : Arrays.asList(realAttrs))
			cols.add(new Attribute(key));
		String[] boolAttrs = {"same_POS"};
		List<String> bool = Arrays.asList("0","1");
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
		ECBDoc doc1 = dataWrapper.docs.get(ev1.file.getName());
		ECBDoc doc2 = dataWrapper.docs.get(ev2.file.getName());
		
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
		@SuppressWarnings("unused")
		DistanceMeasure euclDist = new EuclideanDistance();
//		double sim = 1 - euclDist.compute(tfidf.makeEvVector(ev1Text, ev1Sentences), tfidf.makeEvVector(ev2Text, ev2Sentences));
		double sim = Transforms.cosineSim(this.doubleArrToNDArr(tfidf.makeEvVector(ev1Text, ev1Sentences)),
										 this.doubleArrToNDArr(tfidf.makeEvVector(ev2Text, ev2Sentences)));
		instance.setValue(this.data.attribute("word_distribution_sim"), sim);

		String s_id1 = doc1.toks.get(doc1.mentionIdToTokens.get(ev1.m_id).get(0)).get("sentence");
		String s_id2 = doc2.toks.get(doc2.mentionIdToTokens.get(ev2.m_id).get(0)).get("sentence");
		if(ev1.file.equals(ev2.file) && s_id1.equals(s_id2)) {
			instance.setValue(this.data.attribute("ev_sentence_sim"), 1.0);
		}
		else {
//			sim =  1 - euclDist.compute(tfidf.makeSentVector(ev1Sentences, docCorpus),tfidf.makeSentVector(ev2Sentences, docCorpus));
			sim = Transforms.cosineSim(this.doubleArrToNDArr(tfidf.makeSentVector(ev1Sentences, docCorpus)), 
										this.doubleArrToNDArr(tfidf.makeSentVector(ev2Sentences, docCorpus)));
			instance.setValue(this.data.attribute("ev_sentence_sim"), sim);
		}
		
		/*
		 * avg. head lemma w2v distance
		 */
		String delim = "MAKE_LIST";
		String request = String.format("%1$s/?f=%2$s&p1=%3$s&p2=%4$s", Globals.W2V_SERVER, 
										"n_similarity", 
										delim + doc1.getHeadText(ev1.m_id), 
										delim + doc2.getHeadText(ev2.m_id));
		instance.setValue(this.data.attribute("head_word_vec_sim"), Double.parseDouble(getHTML(request)));
		
		/*
		 * pos
		 */
		HashSet<String> pos1 = posSet(ev1Text);
		HashSet<String> pos2 = posSet(ev2Text);
		boolean NN = pos1.contains("N") && pos2.contains("N");
		boolean VV = pos1.contains("V") && pos2.contains("V");
		instance.setValue(this.data.attribute("same_POS"), (NN || VV) ? "1" : "0");
		
		/*
		 * sentence dist
		 */
//		int ev1_snum = Integer.parseInt(doc1.toks.get(doc1.mentionIdToTokens.get(ev1.m_id).get(0)).get("sentence"));
//		int doc1_numsent = Integer.parseInt(doc1.toks.get(doc1.inOrderToks.get(doc1.inOrderToks.size() - 1)).get("sentence"));
//		double ev1_position = (1.0*ev1_snum)/doc1_numsent;
//		int ev2_snum = Integer.parseInt(doc2.toks.get(doc2.mentionIdToTokens.get(ev2.m_id).get(0)).get("sentence"));
//		int doc2_numsent = Integer.parseInt(doc2.toks.get(doc2.inOrderToks.get(doc2.inOrderToks.size() - 1)).get("sentence"));
//		double ev2_position = (1.0*ev2_snum)/doc2_numsent;
//		instance.setValue(this.data.attribute("sentence_dist"), Math.abs(ev1_position - ev2_position));
			
		
		instance.setValue(this.data.attribute("class"), ev1.corefers(ev2) ? "1" : "0");
		
		
		
		return instance;
	}
	
	private NDArray doubleArrToNDArr(double[] v) {
		float[] vec = new float[v.length];
		for(int i =0 ; i < v.length; i++)
			vec[i] = (float) v[i];
		return new NDArray(vec);
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
	public static List<IndexedWord> mainEvText(EventNode ev, ECBDoc doc){
		List<IndexedWord> text = new LinkedList<IndexedWord>();

		for(String level : doc.mIdToEventText.get(ev.m_id).keySet()) {
			for(int tok_idx : doc.mIdToEventText.get(ev.m_id).get(level).navigableKeySet())
				text.add(doc.mIdToEventText.get(ev.m_id).get(level).get(tok_idx));
		}
		
		return text;
	}
	
	
	/**
	 * get sentence event appears in
	 * @param ev
	 * @param set
	 * @return
	 */
	public static CoreSentence evMainSentence(EventNode ev, ECBDoc doc){
		int s_id = doc.mIdToEventText.get(ev.m_id).get("trigger").firstEntry().getValue().sentIndex();

		return doc.coreDoc.sentences().get(s_id);
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
