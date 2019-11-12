package clustering;

import static org.junit.Assert.assertEquals;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map.Entry;
import java.util.logging.Logger;

import org.apache.avro.mapred.Pair;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;

import com.google.common.graph.GraphBuilder;
import com.google.common.graph.Graphs;
import com.google.common.graph.MutableGraph;
import com.google.common.math.DoubleMath;
import com.google.common.math.Quantiles;
import com.google.common.math.Stats;

import java.util.TreeMap;

import common.ECBWrapper;
import common.Globals;
import common.Main;
import comparer.EvPairDataset;
import naf.EventNode;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;


public class EquivCluster {
	static final Logger LOGGER = Logger.getLogger(EquivCluster.class.getName());
	
	Classifier clf;
	EvPairDataset vectorizer;
	ECBWrapper dataWrapper;
	Instances data;
	
	public EquivCluster(Classifier clf, EvPairDataset vectorizer, ECBWrapper dataWrapper, Instances data) {
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
//			System.out.println(missingEvs.size());
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
		int totalNodes = remainingNodes.size();
		Collections.sort(preds, new CustomComparator());
		
		MutableGraph<EventNode> graph = GraphBuilder.undirected().<EventNode>build();
		for(PairPred pred : preds) {
//			pred.sim > cutoff
			if(pred.sim > cutoff) {
				graph.putEdge(pred.pair.get(0), pred.pair.get(1));
				remainingNodes.removeAll(pred.pair); // remove nodes that are already assigned to a cluster
			}
//			if(graph.nodes().size() == totalNodes)
//				break;
		}
		ArrayList<HashSet<EventNode>> chains = this.dataWrapper.buildCorefChains(graph);
//		SummaryStatistics bestScores = new SummaryStatistics();
		if(remainingNodes.size() > 0 || singletons.size() > 0 || missingEvs.size() > 0) {
			remainingNodes.addAll(singletons);
			remainingNodes.addAll(missingEvs);
//			chains.add(new HashSet<EventNode>(remainingNodes));
			for(EventNode n : remainingNodes)
				chains.add(new HashSet<EventNode>(Arrays.asList(n)));
//			for(EventNode n : remainingNodes) {
//				graph.addNode(n);
//				int bestMatchIdx = -1;
//				double bestScore = Double.MIN_VALUE;
//				for(int i = 0; i < chains.size(); i++) {
//					List<Double> scores = new LinkedList<Double>();
//					for(EventNode chainNode : chains.get(i)) {
//						if(predLog.containsKey(new HashSet<EventNode>(Arrays.asList(n, chainNode))))
//							scores.add(predLog.get(new HashSet<EventNode>(Arrays.asList(n, chainNode))));
//						else { //singleton
//							Instance inst = this.vectorizer.evPairVector(dataWrapper, Arrays.asList(n, chainNode), 
//																		 Globals.LEMMATIZE, Globals.POS);
//							inst.setDataset(this.data);
//							try {
//								double score = this.clf.distributionForInstance(inst)[1];
//								scores.add(score);
//								predLog.put(new HashSet<EventNode>(Arrays.asList(n, chainNode)), score);
//							} catch (Exception e) {
//								e.printStackTrace();
//							}
//							
//						}
//					}
//					if(scores.size() > 0) {
//						double q = Quantiles.percentiles().index(25).compute(scores);
//						if(q > cutoff) {
//							bestMatchIdx = i;
//							bestScore = q;
//						}
//					}
//				}
////				bestScores.addValue(bestScore);
////				if(bestMatchIdx == -1)
////					chains.add(new HashSet<EventNode>(Arrays.asList(n)));
////				else
////					chains.get(bestMatchIdx).add(n);
//				if(bestMatchIdx != -1)
//					chains.get(bestMatchIdx).add(n);
//			}
		}

//		assertEquals(graph.nodes().size(), totalNodes); // TODO: this is not passing
//		System.out.println("avg best ----> " + bestScores.getMean() + ", std: " + bestScores.getStandardDeviation());
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
