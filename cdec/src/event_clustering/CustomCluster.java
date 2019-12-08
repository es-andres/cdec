package event_clustering;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.logging.Logger;

import com.google.common.graph.GraphBuilder;
import com.google.common.graph.MutableGraph;

import ecb_utils.ECBWrapper;
import ecb_utils.EventNode;
import feature_extraction.EvPairDataset;

import weka.classifiers.Classifier;
import weka.core.Instances;


public class CustomCluster {
	static final Logger LOGGER = Logger.getLogger(CustomCluster.class.getName());
	
	Classifier clf;
	EvPairDataset vectorizer;
	ECBWrapper dataWrapper;
	Instances data;
	
	public CustomCluster(Classifier clf, EvPairDataset vectorizer, ECBWrapper dataWrapper, Instances data) {
		this.clf = clf;
		this.vectorizer = vectorizer;
		this.dataWrapper = dataWrapper;
		this.data = data;
	}
	
	public ArrayList<HashSet<EventNode>> cluster(HashMap<HashSet<EventNode>, Double> predLog, 
												 double cutoff, String[] docs, HashSet<EventNode> allMissingEvs) {
		
		LOGGER.info("Doing CDEC");
		
		HashSet<String> inClusterDocs = new HashSet<String>(Arrays.asList(docs));
		ArrayList<PairPred> preds = new ArrayList<PairPred>();
		HashSet<EventNode> remainingNodes = new HashSet<EventNode>(); // used below
		HashSet<EventNode> singletons = new HashSet<EventNode>();
		HashSet<EventNode> missingEvs = new HashSet<EventNode>();
		if(!allMissingEvs.isEmpty()) {
			for(EventNode ev : allMissingEvs) {
				if(inClusterDocs.contains(ev.file.getName()))
					missingEvs.add(ev);
			}
		}
			
		/*
		 * get only pairs that are within current document cluster
		 */

		for(HashSet<EventNode> evPair : predLog.keySet()) {
			List<EventNode> pair = new LinkedList<EventNode>(evPair);
			if(pair.size() == 2) {
				remainingNodes.addAll(pair); // all nodes in this cluster logged at the end of this loop
				PairPred pred = new PairPred(predLog.get(evPair), pair);
				preds.add(pred);
			}
			
		}
		Collections.sort(preds, new CustomComparator());
		System.out.println("CUTOFF: " + cutoff);
		MutableGraph<EventNode> graph = GraphBuilder.undirected().<EventNode>build();
		for(PairPred pred : preds) {
			if(pred.sim > cutoff) {
				graph.putEdge(pred.pair.get(0), pred.pair.get(1));
				remainingNodes.removeAll(pred.pair); // remove nodes that are already assigned to a cluster
			}
		}
		ArrayList<HashSet<EventNode>> chains = this.dataWrapper.buildCorefChains(graph);
		
		if(remainingNodes.size() > 0 || singletons.size() > 0 || missingEvs.size() > 0) {
			remainingNodes.addAll(singletons);
			remainingNodes.addAll(missingEvs);
			for(EventNode n : remainingNodes)
				chains.add(new HashSet<EventNode>(Arrays.asList(n)));
		}

		return chains;
	}
	
	public class PairPred {
		public double sim;
		List<EventNode> pair;
		public PairPred(double sim, List<EventNode> pair) {
			this.sim = sim;
			this.pair = pair;
		}
	}
	public class CustomComparator implements Comparator<PairPred> {

		@Override
		public int compare(PairPred arg0, PairPred arg1) {
			
			return new Double(arg1.sim).compareTo(arg0.sim);
		}
	}
}
