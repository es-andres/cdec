package ecb_utils;

import static org.junit.Assert.assertEquals;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.logging.Logger;

import org.apache.commons.math3.ml.distance.DistanceMeasure;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.deeplearning4j.bagofwords.vectorizer.TfidfVectorizer;

import org.deeplearning4j.text.sentenceiterator.CollectionSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.DefaultTokenizer;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.NGramTokenizerFactory;

import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

import com.google.common.collect.Sets;
import com.google.common.graph.EndpointPair;
import com.google.common.graph.GraphBuilder;
import com.google.common.graph.Graphs;
import com.google.common.graph.MutableGraph;
import com.google.common.math.Quantiles;

import common.GeneralTuple;
import common.Globals;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.pipeline.CoreSentence;
import feature_extraction.EvPairDataset;
import feature_extraction.TFIDF;
import me.tongfei.progressbar.ProgressBar;

public class ECBWrapper {
	private static final Logger LOGGER = Logger.getLogger(ECBWrapper.class.getName());
	
	public MutableGraph<EventNode> trainCorefGraph;
	public MutableGraph<EventNode> testCorefGraph;
	public ArrayList<HashSet<EventNode>> trainCorefChains;
	public ArrayList<HashSet<EventNode>> testCorefChains;
	public HashMap<String, ArrayList<String>> actionToMentions;
	public HashMap<Integer, HashSet<String>> topicToActionSet;
	public List<Integer> trainTopics;
	public List<Integer> testTopics;
	public final List<Integer> DEAD_TOPICS = Arrays.asList(15, 17);
	public HashMap<String, ECBDoc> docs;
	public List<File> files;
	public TfidfVectorizer tfidf;
	public HashSet<EventNode> missingTestEvs;
	
	public ECBWrapper(List<Integer> topics) {
		this.actionToMentions = new HashMap<String, ArrayList<String>>(); // updated in "incorporateChains()"
		this.topicToActionSet = new HashMap<Integer, HashSet<String>>(); // updated in "incorporateChains()"
		this.missingTestEvs = new HashSet<EventNode>();
		this.docs = new HashMap<String, ECBDoc>();
		
		this.files = getFilesFromTopics(topics);
		this.loadFiles(this.files);
		this.tfidf = this.doTFIDF();
	}
	private TfidfVectorizer doTFIDF() {
		List<String> docText = new LinkedList<String>();
		for(File f : this.files) {
			docText.add(this.docs.get(f.getName()).getDocText());
		}
		SentenceIterator docs = new CollectionSentenceIterator(docText);
		CommonPreprocessor tokProc = new CommonPreprocessor();
		Tokenizer tokenize = new DefaultTokenizer(" ");
		tokenize.setTokenPreProcessor(tokProc);
		DefaultTokenizerFactory fact = new DefaultTokenizerFactory();
		fact.setTokenPreProcessor(tokProc);
		NGramTokenizerFactory factory = new NGramTokenizerFactory(fact, 1, 1);
	
        TfidfVectorizer vectorizer = new TfidfVectorizer.Builder()
                .setMinWordFrequency(1)
                .setStopWords(new ArrayList<String>())
                .setTokenizerFactory(factory)
                .setIterator(docs)
//                .labels(labels)
//                .cleanup(true)
                .build();
        
        vectorizer.fit();
        
        for(File f : this.files) {
        	float[] vec = new float[vectorizer.getVocabCache().tokens().size()];
        	String[] words = this.docs.get(f.getName()).getDocText().split(" ");
        	for(String word : words) {
        		int count = vectorizer.getVocabCache().wordFrequency(word);
        		vec[vectorizer.getVocabCache().indexOf(word)] = (float)vectorizer.tfidfWord(word, count, words.length);
        	}

        	this.docs.get(f.getName()).tfidfVec = new NDArray(vec);
        }
        return vectorizer;
	}
	
	public static boolean coferer(ECBDoc doc1, ECBDoc doc2) {
		Set<String> doc1Evs = doc1.actionToMentions.keySet();
		Set<String> doc2Evs = doc2.actionToMentions.keySet();
		doc1Evs.retainAll(doc2Evs);
		if(doc1Evs.size() > 0)
			return true;
		else
			return false;
	}
	public ECBDoc getDoc(File f) {
		return this.docs.get(f.getName());
	}
//	public List<EventNode> getEventSet(List<List<EventNode>> evs, String[] f) {
////		HashSet<String> files = new HashSet<String>(Arrays.asList(f));
////		HashSet<EventNode> evSet = new HashSet<EventNode>();
////		for(List<EventNode> pair : evs) {
////			if(files.contains(pair.get(0).file.getName()) && files.contains(pair.get(0).file.getName())) {
////				evSet.add(pair.get(0));
////				evSet.add(pair.get(1));
////			}
////		}
//		HashSet<Integer> tops = new HashSet<Integer>();
//		for(String fName : f) {
//			tops.add(Integer.parseInt(fName.split("_")[0]));
//		}
//		List<Integer> topics = new LinkedList<Integer>(tops);
//			
//		List<List<EventNode>> evPairs = this.buildBalancedEvPairSet(topics, this.trainCorefGraph);
//		HashSet<EventNode> events = new HashSet<EventNode>();
//		for(List<EventNode> pair : evPairs)
//			events.addAll(pair);
//		//return new ArrayList<EventNode>(evSet)
//		return new ArrayList<EventNode>(events);
//	}
	/**
	 * 
	 * @param topics: list of integers representing an ECB+ topic cluster
	 * @return list of File objects pointing to the corresponding files
	 */
	public static List<File> getFilesFromTopics(List<Integer> topics) {

		LinkedList<File> files = new LinkedList<File>();
        for (final File f : Globals.ECBPLUS_DIR.toFile().listFiles()) {
            if (f.isFile() && Globals.cleanSentences.containsKey(ECBWrapper.cleanFileName(f))) {
            	String topicNum = f.getName().split("_")[0];
            	
            	if(topics.contains(Integer.parseInt(topicNum)))
                	files.add(f);
            }
        }
		return files;
	}
	
	private void loadFiles(List<File> files) {

		for(File f : ProgressBar.wrap(files, "Load ECB files")) {
			ECBDoc doc = new ECBDoc(f);
			this.incorporateChains(doc);
			this.docs.put(f.getName(),  doc);
		}
	}
	
	private void incorporateChains(ECBDoc doc) {
		
		for(String ev_id : doc.actionToMentions.keySet()) {
			/*
			 * map ev_id to topic
			 */
			if(!this.topicToActionSet.containsKey(doc.topic))
				this.topicToActionSet.put(doc.topic, new HashSet<String>());
			this.topicToActionSet.get(doc.topic).add(ev_id);
			/*
			 * incorporate doc coref chains into global chains
			 */
			if(!this.actionToMentions.containsKey(ev_id))
				this.actionToMentions.put(ev_id, new ArrayList<String>());
			for(String m_id : doc.actionToMentions.get(ev_id)) {
				String globalMidKey = doc.file.getName() + Globals.DELIM + m_id;
				// Some mentions appear more than once because they span
				// more than one token. Only add each mention once.
				if(!this.actionToMentions.get(ev_id).contains(globalMidKey))
					this.actionToMentions.get(ev_id).add(globalMidKey);
			}
		}
	}
	
	public List<List<EventNode>> getGoldDocClusteringEvPairs(List<Integer> topics, HashMap<String, String> fToClustMap){
		MutableGraph<EventNode> graph = this.buildCorefGraph(topics, fToClustMap);
		List<List<EventNode>> pairs = new LinkedList<List<EventNode>>();
		HashSet<HashSet<EventNode>> seen = new HashSet<HashSet<EventNode>>();
		for(EventNode ev1 : graph.nodes()) {
			for(EventNode ev2 : graph.nodes()) {
				if(!ev1.equals(ev2) 
						&& fToClustMap.get(ev1.file.getName()).equals(fToClustMap.get(ev2.file.getName()))
						&& seen.add(new HashSet<EventNode>(Arrays.asList(ev1, ev2))))
					pairs.add(Arrays.asList(ev1,ev2));
			}
		}
		return pairs;
	}
	public static HashMap<String, Double> getLabelDistribution(List<List<EventNode>> evs) {
		double truePairs = 0;
		double falsePairs = 0;
		for(List<EventNode> ev : evs) {
			if(ev.get(0).corefers(ev.get(1)))
				truePairs++;
			else
				falsePairs++;
		}
		HashMap<String, Double> res = new HashMap<String, Double>();
		res.put("true", truePairs);
		res.put("false", falsePairs);

		return res;
	}
	public List<List<EventNode>> buildTrainPairs(List<Integer> topics, boolean balanced, HashMap<String, String> fToClustMap) {
		LOGGER.info("Building train set");
		this.trainTopics = topics;
		this.trainCorefGraph = this.buildCorefGraph(topics, fToClustMap);
		this.trainCorefChains = this.buildCorefChains(this.trainCorefGraph);
//		IntSummaryStatistics statistics = Globals.globalCorefChains.get(set).stream()
//      .mapToInt(HashSet<EventNode>::size)
//      .summaryStatistics();
		if(balanced)
			return this.buildBalancedEvPairSet(topics, this.trainCorefGraph, fToClustMap, 3);
		else {
			List<List<EventNode>> evs = new LinkedList<List<EventNode>>();
			HashSet<HashSet<EventNode>> seen = new HashSet<HashSet<EventNode>>();
			for(EventNode ev1 : ProgressBar.wrap(this.trainCorefGraph.nodes(),"building train pairs")) {
				for(EventNode ev2 : this.trainCorefGraph.nodes()) {
					if(!ev1.equals(ev2) && ev1.getTopic().equals(ev2.getTopic()) 
							&& ev1.getSubTopic().equals(ev2.getSubTopic())
							&& seen.add(new HashSet<EventNode>(Arrays.asList(ev1,ev2)))) {
						evs.add(Arrays.asList(ev1, ev2));
					}
				}
			}
			return evs;
		}
	}
	
	// not true/false balanced
	public List<List<EventNode>> buildTestPairs(List<Integer> topics, HashMap<String, String> fNameToClustMap, 
												double evDistCutoff, double docSimCutoff, boolean filter, boolean balanced) {
		LOGGER.info("Building test set");
		this.testTopics = topics;
		this.testCorefGraph = this.buildCorefGraph(topics, fNameToClustMap);
		this.testCorefChains = this.buildCorefChains(this.testCorefGraph);
		
		if(balanced)
			return this.buildBalancedEvPairSet(topics, this.testCorefGraph, fNameToClustMap, 1);
		else {
			List<List<EventNode>> evs = new LinkedList<List<EventNode>>();
			HashSet<EventNode> missingEvs = new HashSet<EventNode>(this.testCorefGraph.nodes());
			Set<Set<EventNode>> pairs = Sets.combinations(this.testCorefGraph.nodes(), 2);
			HashMap<HashSet<String>, Double> filePairToCosSim = null;
			HashMap<String, Double> clusterIdToCutoff = null;
			if(filter) {
				GeneralTuple<HashMap<HashSet<String>, Double>, HashMap<String, Double>>  t = fitTfidfPerDocCluster(fNameToClustMap);
				filePairToCosSim = t.first;
				clusterIdToCutoff = t.second;
			}
			
			for(Set<EventNode> pair : pairs) {
				List<EventNode> pairList = new LinkedList<EventNode>(pair);
				EventNode ev1 = pairList.get(0);
				EventNode ev2 = pairList.get(1);
				if(fNameToClustMap.get(ev1.file.getName()).equals(fNameToClustMap.get(ev2.file.getName()))) {
					if(filter) {
						double docSim = 1;
						if(!ev1.file.getName().equals(ev2.file.getName()))
							docSim = filePairToCosSim.get(new HashSet<String>(Arrays.asList(ev1.file.getName(), ev2.file.getName())));
						
						double evDist = this.evTextDist(ev1, ev2);
						if(docSim > clusterIdToCutoff.get(fNameToClustMap.get(ev1.file.getName()))
								&& evDist < evDistCutoff) {
							missingEvs.removeAll(Arrays.asList(ev1, ev2));
							evs.add(Arrays.asList(ev1, ev2));
						}
					}
					else {
						missingEvs.removeAll(Arrays.asList(ev1, ev2));
						evs.add(new LinkedList<EventNode>(pair));
					}
				}
			}
			this.missingTestEvs = missingEvs;
			return evs;
		}
	}
	
	private GeneralTuple<HashMap<HashSet<String>, Double>, HashMap<String, Double>> 
							fitTfidfPerDocCluster(HashMap<String, String> fNameToClustMap) {
		HashMap<String, Double> clusterIdToCutoff = new HashMap<String, Double>();
		HashMap<HashSet<String>, Double> filePairToCosSim = new HashMap<HashSet<String>, Double>();
		HashMap<String, NDArray> fNameToTfidfVec = new HashMap<String, NDArray>();
		HashMap<String, HashSet<String>> clustIdToFiles = new HashMap<String, HashSet<String>>();
		
		/*
		 * inver fname, clustid map
		 */
		for(String fName: fNameToClustMap.keySet()) {
			String clustId = fNameToClustMap.get(fName);
			if(!clustIdToFiles.containsKey(clustId))
				clustIdToFiles.put(clustId, new HashSet<String>());
			clustIdToFiles.get(clustId).add(fName);
		}
		
		/*
		 * fit a tfidf model for each cluster
		 */
		for(String clustId : clustIdToFiles.keySet()) {
			List<String> docText = new LinkedList<String>();
			for(String fName : clustIdToFiles.get(clustId))
				docText.add(this.docs.get(fName).getDocText());
		
			SentenceIterator docs = new CollectionSentenceIterator(docText);
			CommonPreprocessor tokProc = new CommonPreprocessor();
			Tokenizer tokenize = new DefaultTokenizer(" ");
			tokenize.setTokenPreProcessor(tokProc);
			DefaultTokenizerFactory fact = new DefaultTokenizerFactory();
			fact.setTokenPreProcessor(tokProc);
			NGramTokenizerFactory factory = new NGramTokenizerFactory(fact, 1, 1);
		
	        TfidfVectorizer vectorizer = new TfidfVectorizer.Builder()
	                .setMinWordFrequency(1)
	                .setStopWords(new ArrayList<String>())
	                .setTokenizerFactory(factory)
	                .setIterator(docs)
	                .build();
	        
	        vectorizer.fit();
	        
	        /*
	         * get tfidf vector for each document in this cluster
	         */
	        for(String fName : clustIdToFiles.get(clustId)) {
	        	System.out.println(fName);
	        	float[] vec = new float[vectorizer.getVocabCache().tokens().size()];
	        	String[] words = this.docs.get(fName).getDocText().split(" ");
	        	for(String word : words) {
	        		int count = vectorizer.getVocabCache().wordFrequency(word);
	        		vec[vectorizer.getVocabCache().indexOf(word)] = (float)vectorizer.tfidfWord(word, count, words.length);
	        	}

	        	fNameToTfidfVec.put(fName, new NDArray(vec));
	        }
	        
	        /*
	         * calculate pairwise cos sim 
	         */
	        SummaryStatistics cosSims = new SummaryStatistics();
	        LinkedList<Double> cosList = new LinkedList<Double>();
	        for(Set<String> pair : Sets.combinations(clustIdToFiles.get(clustId), 2)) {
	        	List<String> files = new LinkedList<String>(pair);
	        	String f1 = files.get(0);
	        	String f2 = files.get(1);
	        	double sim = Transforms.cosineSim(fNameToTfidfVec.get(f1), fNameToTfidfVec.get(f2));
	        	cosSims.addValue(sim);
	        	cosList.add(sim);
	        	filePairToCosSim.put(new HashSet<String>(pair), sim);
	        }
	        System.out.println(clustId + "(sum) : " + (cosSims.getMean()- 2*cosSims.getVariance()));
	        double q = Quantiles.percentiles().index(25).compute(cosList);
	        System.out.println(clustId + "(q) : " + q);
	        clusterIdToCutoff.put(clustId, q);
		}
		return new GeneralTuple<HashMap<HashSet<String>, Double>, HashMap<String, Double>>(filePairToCosSim, clusterIdToCutoff);
		
	}
	
	private double evTextDist(EventNode ev1, EventNode ev2) {
		
		List<IndexedWord> ev1Text = EvPairDataset.mainEvText(ev1, this.docs.get(ev1.file.getName()));
		List<IndexedWord> ev2Text = EvPairDataset.mainEvText(ev2, this.docs.get(ev2.file.getName()));
		List<CoreSentence> ev1Sentences = Arrays.asList(EvPairDataset.evMainSentence(ev1, this.docs.get(ev1.file.getName())));
		List<CoreSentence> ev2Sentences = Arrays.asList(EvPairDataset.evMainSentence(ev2, this.docs.get(ev2.file.getName())));
		List<CoreSentence> sentenceCorpus = EvPairDataset.makeSentCorpus(ev1Sentences, ev2Sentences, ev1.file.equals(ev2.file));
		TFIDF comparer = new TFIDF(sentenceCorpus, Globals.LEMMATIZE, Globals.POS, 1);
		DistanceMeasure dist = new EuclideanDistance();
		return dist.compute(comparer.makeEvVector(ev1Text, ev1Sentences), comparer.makeEvVector(ev2Text, ev2Sentences));
	}
	
	
	/**
	 * Make a global graph(V,E), where 
	 * V = {ev_1,...,ev_n} s.t. ev_i = event mention 
	 * E = {(ev_i,ev_j)...} s.t. ev_i,ev_j corefer
	 * @param set: train or test
	 * @return none.
	 */
	private MutableGraph<EventNode> buildCorefGraph(List<Integer> topics, HashMap<String, String> fNameToClust) {
		
		MutableGraph<EventNode> graph = GraphBuilder.undirected().<EventNode>build();
		HashSet<EventNode> singletons = new HashSet<EventNode>();
		for(int topic : topics) {
			for(String ev_id : this.topicToActionSet.get(topic)) { // add edges for every pair of corefering events
				ArrayList<String> mIdChain = this.actionToMentions.get(ev_id);
				if(mIdChain.size() == 1) { // singleton event
					String globalMidKey = mIdChain.iterator().next();
					Path file = Paths.get(Globals.ECBPLUS_DIR.toString(), globalMidKey.split(Globals.DELIM)[0]);
					String localMid = globalMidKey.split(Globals.DELIM)[1];
					EventNode ev = new EventNode(file.toFile(), localMid, ev_id); // always size 1
					if(fNameToClust.containsKey(ev.file.getName())) {
						graph.addNode(ev);
						singletons.add(ev);
					}
				}
				else { // add edges 
					for(Set<String> comb : Sets.combinations(new HashSet<String>(mIdChain), 2)) { // get all size 2 combinations
						String[] m_id_pair = comb.toArray(new String[2]); // always size 2
						
						Path fileKey1 = Paths.get(Globals.ECBPLUS_DIR.toString(), m_id_pair[0].split(Globals.DELIM)[0]);
						String localMidKey1= m_id_pair[0].split(Globals.DELIM)[1];
						EventNode ev1 = new EventNode(fileKey1.toFile(), localMidKey1, ev_id); // ev1 in pair
						assertEquals(singletons.contains(ev1), false);
						
						Path fileKey2 = Paths.get(Globals.ECBPLUS_DIR.toString(), m_id_pair[1].split(Globals.DELIM)[0]);
						String localMidKey2= m_id_pair[1].split(Globals.DELIM)[1];
						EventNode ev2 = new EventNode(fileKey2.toFile(), localMidKey2, ev_id); // ev2 in pair
						assertEquals(singletons.contains(ev2), false);
						assertEquals(ev1.corefers(ev2), true);
						if(fNameToClust.get(ev1.file.getName()).equals(fNameToClust.get(ev2.file.getName()))) {
							graph.putEdge(ev1, ev2); // adds nodes silently if they don't exist
						}
//						else if(ev1.corefers(ev2)
//								&& !fNameToClust.get(ev1.file.getName()).equals(fNameToClust.get(ev2.file.getName()))) {
//							System.out.println(" NOT adding .... -> " + ev1 + " ... " + ev2);
//							System.out.println(fNameToClust.get(ev1.file.getName() + " ? " + fNameToClust.get(ev2.file.getName())));
//						}
							
					}
				}
			}
		}
		return graph;
	}
	
	/**
	 * Returns all coreference chains in given coreference graph
	 * @param corefGraph
	 * @return
	 */
	public ArrayList<HashSet<EventNode>> buildCorefChains(MutableGraph<EventNode> corefGraph) {
		HashSet<EventNode> visitedNodes = new HashSet<EventNode>();
		ArrayList<HashSet<EventNode>> chains = new ArrayList<HashSet<EventNode>>();
		for(EventNode node : corefGraph.nodes()) {
			if(!visitedNodes.contains(node)) {
				Set<EventNode> reachableNodes = Graphs.reachableNodes(corefGraph, node);
				visitedNodes.addAll(reachableNodes);
				chains.add(new HashSet<EventNode>(reachableNodes));
			}
		}
		return chains;
	}
	
	
	private List<List<EventNode>> buildBalancedEvPairSet(List<Integer> topics, MutableGraph<EventNode> graph, HashMap<String, String> fToClustMap,
														 int multiplier) {

		/*
		 * Get all coref chains for these topics
		 */
		List<List<EventNode>> truePairs = new ArrayList<List<EventNode>>();
		for(EndpointPair<EventNode> p : graph.edges()) {
			ArrayList<EventNode> thisPair = new ArrayList<EventNode>();
			if(!p.nodeU().equals(p.nodeV())) {
				thisPair.add(p.nodeU());
				thisPair.add(p.nodeV());
				truePairs.add(thisPair);
			}
		}
		HashSet<EventNode> addedNodes = new HashSet<EventNode>();
		for(List<EventNode> pair : truePairs)
			addedNodes.addAll(pair);
		HashSet<EventNode> allNodes = new HashSet<EventNode>(graph.nodes());
		HashSet<EventNode> singletons = new HashSet<EventNode>(Sets.difference(allNodes, addedNodes));
		HashSet<HashSet<EventNode>> falsePairs = new HashSet<HashSet<EventNode>>();
		if(singletons.size() > 1) {
			
			for(Set<EventNode> pair : Sets.combinations(singletons, 2)) {
				if(singletons.size() == 0)
					break;
				falsePairs.add(new HashSet<EventNode>(pair));
				singletons.removeAll(pair);
			}
		}
		else if(singletons.size() == 1)
			falsePairs.add(new HashSet<EventNode>(Arrays.asList(singletons.iterator().next(), addedNodes.iterator().next())));
		
		
		List<String> actionList = new ArrayList<String>();
		for(int t : topics)
			actionList.addAll(this.topicToActionSet.get(t));
		
		Set<Set<EventNode>> combs = Sets.combinations(graph.nodes(), 2);
		for(Set<EventNode> p : combs) {
			if(multiplier != -1) {
				if(falsePairs.size() >= multiplier*truePairs.size())
					break;
			}
			List<EventNode> pair = new LinkedList<EventNode>(p);
			EventNode ev1 = pair.get(0);
			EventNode ev2 = pair.get(1);
			if(!ev1.corefers(ev2)
					&& fToClustMap.get(ev1.file.getName()).equals(fToClustMap.get(ev2.file.getName())))
				falsePairs.add(new HashSet<EventNode>(p));
		}
		
		/*
		 * over sample true pairs for training
		 */
		if(multiplier == -1) {
			Random random = new Random();
//			System.out.println("pre oversampling true pairs: " + truePairs.size());
			while(truePairs.size() < falsePairs.size()) {
				List<EventNode> pair = truePairs.get(random.nextInt(truePairs.size()));
				truePairs.add(pair);
			}
		}
		List<List<EventNode>> pairs = new LinkedList<List<EventNode>>();
		for(List<EventNode> pair : truePairs)
			pairs.add(pair);
		for(HashSet<EventNode> pair : falsePairs)
			pairs.add(new LinkedList<EventNode>(pair));
//		System.out.println("true pairs: " + truePairs.size());
//		System.out.println("false pairs: " + falsePairs.size());
		Collections.shuffle(pairs);
		
		return pairs;
	}
	
	public static String cleanFileName(File file) {
		String[] spl = file.getName().split("_");
		String clean = spl[0] + "_" + spl[1];
		return clean;
	}
	
}
