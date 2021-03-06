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
import java.util.Set;
import java.util.logging.Logger;

import com.google.common.collect.Sets;
import com.google.common.graph.EndpointPair;
import com.google.common.graph.GraphBuilder;
import com.google.common.graph.Graphs;
import com.google.common.graph.MutableGraph;

import common.GeneralTuple;
import common.Globals;

import me.tongfei.progressbar.ProgressBar;

public class ECBWrapper {
	private static final Logger LOGGER = Logger.getLogger(ECBWrapper.class.getName());
	
	public MutableGraph<EventNode> trainCorefGraph;
	public MutableGraph<EventNode> testCorefGraph;
	public ArrayList<HashSet<EventNode>> trainCorefChains;
	public ArrayList<HashSet<EventNode>> testCorefChains;
	public HashMap<String, ArrayList<String>> globalActionToMentions;
	public HashMap<Integer, HashSet<String>> topicToActionSet;
	public List<Integer> trainTopics;
	public List<Integer> testTopics;
	public HashMap<String, ECBDoc> docs;
	public List<File> files;
	public HashSet<EventNode> missingTestEvs;
	
	public ECBWrapper(List<Integer> topics) {
		this.globalActionToMentions = new HashMap<String, ArrayList<String>>(); // updated in "incorporateChains()"
		this.topicToActionSet = new HashMap<Integer, HashSet<String>>(); // updated in "incorporateChains()"
		this.missingTestEvs = new HashSet<EventNode>();
		this.docs = new HashMap<String, ECBDoc>();
		
		this.files = getFilesFromTopics(topics);
		this.loadFiles(this.files);
	}

	
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
	
	public GeneralTuple<List<List<EventNode>>, List<List<EventNode>>> buildTrainAndDevPairs(List<Integer> topics, boolean balanced, HashMap<String, String> fToClustMap) {
		this.trainTopics = topics;
		this.trainCorefGraph = this.buildCorefGraph(topics, fToClustMap);
		this.trainCorefChains = this.buildCorefChains(this.trainCorefGraph);
		int falseTrueRatio = 3;
		return this.buildTrainAndDevSets(topics, this.trainCorefGraph, fToClustMap, falseTrueRatio);
	}
	

	public List<List<EventNode>> buildTestPairs(List<Integer> topics, HashMap<String, String> fNameToClustMap) {
		LOGGER.info("Building test set");
		this.testTopics = topics;
		this.testCorefGraph = this.buildCorefGraph(topics, fNameToClustMap);
		this.testCorefChains = this.buildCorefChains(this.testCorefGraph);

		List<List<EventNode>> evs = new LinkedList<List<EventNode>>();
		Set<EventNode> predEvs = this.testCorefGraph.nodes();
		if(Globals.USE_TEST_PRED_EVS)
			predEvs = this.getPredEvs(fNameToClustMap);
		
		HashSet<EventNode> missingEvs = new HashSet<EventNode>(predEvs);
		Set<Set<EventNode>> pairs = Sets.combinations(predEvs, 2);
		
		for(Set<EventNode> pair : pairs) {
			List<EventNode> pairList = new LinkedList<EventNode>(pair);
			EventNode ev1 = pairList.get(0);
			EventNode ev2 = pairList.get(1);
			if(fNameToClustMap.get(ev1.file.getName()).equals(fNameToClustMap.get(ev2.file.getName()))) {
				missingEvs.removeAll(Arrays.asList(ev1, ev2));
				evs.add(new LinkedList<EventNode>(pair));
			}
		}
		
		this.missingTestEvs = missingEvs;
		return evs;
		
	}
	
	private HashSet<EventNode> getPredEvs(HashMap<String, String> fNameToClustMap){
		
		HashSet<EventNode> predEvs = new HashSet<EventNode>();
		
		int i = 1;
		for(String globalPredKey : this.globalActionToMentions.get("preds")) {
			Path fileKey = Paths.get(Globals.ECBPLUS_DIR.toString(), globalPredKey.split(Globals.DELIM)[0]);
			if(fNameToClustMap.containsKey(fileKey.toFile().getName())) {
				String localMidKey= globalPredKey.split(Globals.DELIM)[1];
				EventNode ev = new EventNode(fileKey.toFile(), localMidKey, "unk_"+i++); // ev1 in pair
				predEvs.add(ev);
			}
		}
		return predEvs;
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
				ArrayList<String> mIdChain = this.globalActionToMentions.get(ev_id);
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
					}
				}
			}
		}
		return graph;
	}
	
	private GeneralTuple<List<List<EventNode>>, List<List<EventNode>>> buildTrainAndDevSets(List<Integer> topics, MutableGraph<EventNode> graph, HashMap<String, String> fToClustMap,
														 int multiplier) {

		HashSet<EventNode> addedEvs = new HashSet<EventNode>();
		List<List<EventNode>> truePairs = new ArrayList<List<EventNode>>();
		HashSet<HashSet<EventNode>> falsePairs = new HashSet<HashSet<EventNode>>();
		
		/*
		 * TRUE: get corefering event pairs, log added events
		 */
		
		for(EndpointPair<EventNode> p : graph.edges()) {
			if(!p.nodeU().equals(p.nodeV())) {
				truePairs.add(Arrays.asList(p.nodeU(), p.nodeV()));
				addedEvs.addAll(Arrays.asList(p.nodeU(), p.nodeV()));
			}
		}
		/*
		 * singleton events do not participate in an edge, so add them here
		 */
		HashSet<EventNode> singletons = new HashSet<EventNode>(Sets.difference(graph.nodes(), addedEvs));
		
		/*
		 * FALSE: add non-corefering event pairs, beginning with singletons
		 */
		
		if(singletons.size() > 1) {
			for(Set<EventNode> pair : Sets.combinations(singletons, 2)) {
				if(singletons.size() == 0)
					break;
				falsePairs.add(new HashSet<EventNode>(pair));
				singletons.removeAll(pair);
			}
		}
		else if(singletons.size() == 1)
			falsePairs.add(new HashSet<EventNode>(Arrays.asList(singletons.iterator().next(), addedEvs.iterator().next())));
		
	
		/*
		 * add remaining (within doc-cluster) false pairs
		 */
		Set<Set<EventNode>> combs = Sets.combinations(graph.nodes(), 2);
		for(Set<EventNode> p : combs) {
			if(multiplier != -1) {
				if(falsePairs.size() >= multiplier*truePairs.size())
					break;
			}
			List<EventNode> pair = new LinkedList<EventNode>(p);
			EventNode ev1 = pair.get(0);
			EventNode ev2 = pair.get(1);
			if(!ev1.corefers(ev2) && !falsePairs.contains(p)
					&& fToClustMap.get(ev1.file.getName()).equals(fToClustMap.get(ev2.file.getName())))
				falsePairs.add(new HashSet<EventNode>(p));
		}
		List<List<EventNode>> falsePairsList = new LinkedList<List<EventNode>>();
		for(HashSet<EventNode> pair : falsePairs)
			falsePairsList.add(new LinkedList<EventNode>(pair));

		Collections.shuffle(truePairs);
		Collections.shuffle(falsePairsList);
		
		/*
		 * split randomly into train and dev
		 */
		int totalPairs = truePairs.size() + falsePairsList.size();
		int dev_true_pairs = (int)(totalPairs*0.01);
		int dev_false_pairs = (int)(totalPairs*0.15);
		
		List<List<EventNode>> train = new LinkedList<List<EventNode>>();
		List<List<EventNode>> dev = new LinkedList<List<EventNode>>();
		
		/*
		 * true pairs
		 */
		for(List<EventNode> pair : truePairs) {
			if(dev_true_pairs > 0) {
				dev.add(pair);
				dev_true_pairs--;
			}
			else 
				train.add(pair);
		}
		
		/*
		 * false pairs
		 */
		for(List<EventNode> pair : falsePairsList) {
			if(dev_false_pairs > 0) {
				dev.add(pair);
				dev_false_pairs--;
			}
			else
				train.add(pair);
		}
		
		LOGGER.info("train: " + train.size());
		LOGGER.info("dev: " + dev.size());
		
		return new GeneralTuple<List<List<EventNode>>, List<List<EventNode>>>(train, dev);
	}
	
	private void incorporateChains(ECBDoc doc) {
		
		for(String ev_id : doc.actionToMentions.keySet()) {

			/*
			 * map gold ev_id to topic
				Collections.shuffle(train);
		Collections.shuffle(dev);	 */
			if(!ev_id.equals("preds")) {
				if(!this.topicToActionSet.containsKey(doc.topic))
					this.topicToActionSet.put(doc.topic, new HashSet<String>());
				this.topicToActionSet.get(doc.topic).add(ev_id);
			}
			/*
			 * incorporate doc coref chains into global chains
			 */
			if(!this.globalActionToMentions.containsKey(ev_id))
				this.globalActionToMentions.put(ev_id, new ArrayList<String>());
			
			for(String m_id : doc.actionToMentions.get(ev_id)) {
				String globalMidKey = doc.file.getName() + Globals.DELIM + m_id;
				// Some mentions appear more than once because they span
				// more than one token. Only add each mention once.
				if(!this.globalActionToMentions.get(ev_id).contains(globalMidKey))
					this.globalActionToMentions.get(ev_id).add(globalMidKey);
			}
		}
	}
	
	public List<List<EventNode>> getGoldDocClusteringEvPairs(List<Integer> topics, HashMap<String, String> fToClustMap){
		MutableGraph<EventNode> graph = this.buildCorefGraph(topics, fToClustMap);
		List<List<EventNode>> pairs = new LinkedList<List<EventNode>>();

		for(Set<EventNode> p : Sets.combinations(graph.nodes(), 2)) {
			List<EventNode> pair = new LinkedList<EventNode>(p);
			EventNode ev1 = pair.get(0);
			EventNode ev2 = pair.get(1);
			if(!ev1.equals(ev2) 
					&& fToClustMap.get(ev1.file.getName()).equals(fToClustMap.get(ev2.file.getName())))
				pairs.add(Arrays.asList(ev1,ev2));
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
	
	public static String cleanFileName(File file) {
		String[] spl = file.getName().split("_");
		String clean = spl[0] + "_" + spl[1];
		return clean;
	}
	
}
