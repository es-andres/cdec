package main;

import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVSaver;

import java.io.*;
import java.util.*;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import com.google.common.base.Functions;
import com.google.common.base.Splitter;
import com.google.common.collect.Lists;

import common.GeneralTuple;
import common.Globals;
import common.SerUtils;
import common.Utils;
import ecb_utils.ECBWrapper;
import ecb_utils.EventNode;
import event_clustering.CustomCluster;
import feature_extraction.EvPairDataset;
import result_calculation.ConllFile;
import result_calculation.PerformanceMetric;


//@SuppressWarnings("unused")
public class Main {
	
	static final Logger LOGGER = Logger.getLogger(Main.class.getName());
	
	/*
	 * note: word embedding server must be started before running this (see README)
	 */
	public static void main(String[] args) throws IOException
	{
		int k = 5;
		double beta = 0.8;
		double doc_clust_damping = 0.5;
		String n_doc_clusts = "";
		String logMessage = String.format("cdec, %s-cv", k);

		boolean useGoldDocClusters = false;
		
		if(!Globals.CACHED_CORE_DOCS.toFile().exists())
			SerUtils.cacheSemanticParses();
		
		/*
		 * [0] Prepare cross validation folds
		 */
		// make list of topics for CV fold generation
		List<Integer> topics = IntStream.rangeClosed(1,45).boxed().collect(Collectors.toList());
		topics.removeAll(Globals.DEAD_TOPICS); // some topics numbers don't exist in ECB+
		ECBWrapper dataWrapper = new ECBWrapper(topics); // loads all documents, including corenlp parses

		
		boolean shuffle = true;
		LinkedList<List<Integer>> kTestSets = Utils.cvFolds(topics, k, shuffle); // cv
		
    	String[] conllMetrics = {"muc", "bcub", "ceafm", "ceafe", "blanc"};
    	String[] types = {"f1", "recall", "precision"};
		HashMap<String, ArrayList<Double>> scores = Utils.initScoreMap(conllMetrics, types); // performance scores stored here
		/*
		 * 0.1 Run CDEC procedure on each fold
		 */
		LOGGER.info(String.format("\nRunning %s-fold CV\n", k));
		int foldNum = 1;
		for (List<Integer> testTopics : kTestSets) {
			scores.get("beta").add(beta);
			LOGGER.info(String.format("Fold %s -> Test set: %s", foldNum++, testTopics));
			
				
			/*
		     * [1] Document clustering on test documents
		     */
			
		    /*
		     * Call python script to cluster documents
		     */
			Process proc = new ProcessBuilder("bash", 
											  Globals.PY_DOC_CLUSTER.toString(), 
											  String.join(" ", Lists.transform(testTopics, Functions.toStringFunction())),
											  String.valueOf(doc_clust_damping),
											  n_doc_clusts).start();
			BufferedReader output = new BufferedReader(new InputStreamReader(proc.getInputStream()));
			String[] s = output.readLine().split("BREAK");
			Map<String, String>  testPredDocClusters = Splitter.on(",").withKeyValueSeparator(":").split(s[0].replace(" '", "").replace("'", ""));
			Map<String, String>  docClusterPerf = Splitter.on(",").withKeyValueSeparator(":").split(s[1].replace(" '", "").replace("'",""));
			
			for(String key : docClusterPerf.keySet())
				scores.get(key).add(Double.parseDouble(docClusterPerf.get(key)));
			
		    /*
		     * [2] Train pairwise event coreference classifier
		     */
			
			/*
			 * 2.1 Build event-pair train set, using it to compute cutoffs
			 */
		    List<Integer> trainTopics = topics.stream().filter(t -> !testTopics.contains(t)).collect(Collectors.toList());
		    boolean balancedTrain = true; 
		    HashMap<String, String> trainfNameToGoldDocClustMap = Utils.fNameToGoldDocCluster(trainTopics, dataWrapper);
		    GeneralTuple<List<List<EventNode>>, List<List<EventNode>>> trainAndDevPairs 
		    							= dataWrapper.buildTrainAndDevPairs(trainTopics, balancedTrain, trainfNameToGoldDocClustMap);
		    List<List<EventNode>> trainPairs = trainAndDevPairs.first;
		    List<List<EventNode>> devPairs = trainAndDevPairs.second;
		    
		    /*
		     * 2.2 Build event-pair test set
		     */
		    HashMap<String, String> testFNameToDocClustMap = null;
		    if(useGoldDocClusters)
		    	testFNameToDocClustMap = Utils.fNameToGoldDocCluster(testTopics, dataWrapper);
		    else
		    	testFNameToDocClustMap = Utils.fNameToPredictedDocCluster(testPredDocClusters); // predicted doc clusters

			List<List<EventNode>> testPairs = dataWrapper.buildTestPairs(testTopics, testFNameToDocClustMap);
			
		    /*
		     * 2.3 log number of true/false in filtered dataset using doc clusters and gold dataset
		     */
			// with doc clustering
		    HashMap<String, Double> labelDist = ECBWrapper.getLabelDistribution(dataWrapper.buildTestPairs(testTopics, testFNameToDocClustMap));
		    // without doc clustering
		    HashMap<String, String> testFNameToGoldDocClust = Utils.fNameToGoldDocCluster(testTopics, dataWrapper);
		    labelDist = ECBWrapper.getLabelDistribution(dataWrapper.getGoldDocClusteringEvPairs(testTopics, testFNameToGoldDocClust)); // did not check for equality, bug here
		    scores.get("gold_true_pairs").add(labelDist.get("true"));
		    scores.get("gold_false_pairs").add(labelDist.get("false"));
		    System.out.println("gold: " + labelDist);
		    // current experiment
		    labelDist = ECBWrapper.getLabelDistribution(testPairs);
		    scores.get("this_true_pairs").add(labelDist.get("true"));
		    scores.get("this_false_pairs").add(labelDist.get("false"));
		    System.out.println("this: " + labelDist);
		    labelDist = ECBWrapper.getLabelDistribution(trainPairs);
		    scores.get("train_true_pairs").add(labelDist.get("true"));
		    scores.get("train_false_pairs").add(labelDist.get("false"));
		    System.out.println("train: " + labelDist);
		    
		    /*
		     * 2.4 Extract features from training pairs to train classifier
		fNameToGoldDocCluster     */
			EvPairDataset dataMaker = new EvPairDataset();
		    Instances train = dataMaker.makeDataset(dataWrapper, trainPairs, Globals.LEMMATIZE, Globals.POS);
//		    Utils.saveCsv(train); // save training set as csv

		    LinkedList<GeneralTuple<Instance, List<EventNode>>> dev = new LinkedList<GeneralTuple<Instance, List<EventNode>>>();
		    for(List<EventNode> pair : devPairs) {
				Instance inst = dataMaker.evPairVector(dataWrapper, pair, Globals.LEMMATIZE, Globals.POS);
				inst.setDataset(train);
				dev.add(new GeneralTuple<Instance, List<EventNode>> (inst, pair));
			}
			LinkedList<GeneralTuple<Instance, List<EventNode>>> test = new LinkedList<GeneralTuple<Instance, List<EventNode>>>();
		    for(List<EventNode> pair : testPairs) {
				Instance inst = dataMaker.evPairVector(dataWrapper, pair, Globals.LEMMATIZE, Globals.POS);
				inst.setDataset(train);
				test.add(new GeneralTuple<Instance, List<EventNode>> (inst, pair));
			}

		    /*
		     * 2.5 Train ev-pair classifier
		     */
		    MultilayerPerceptron clf = new MultilayerPerceptron();
		    clf.setLearningRate(0.0001); 
		    clf.setHiddenLayers("a");
		    clf.setMomentum(0.85);

		    try {
		    	LOGGER.info("Traning classifier");
				clf.buildClassifier(train);
			} 
		    catch (Exception e) {
				e.printStackTrace();
			}

			/*
			 * [3] CDEC on document clusters
			 */
			
		    /*
		     * 3.1 Test classifier and log predictions 
		     * (so they don't have to be recomputed)
		     */

		    GeneralTuple<Double, HashMap<HashSet<EventNode>, Double>> res = Utils.testClassifier(clf, test, dev, scores,
		    																					 beta);

		    double predictionCutoff = res.first;
		    HashMap<HashSet<EventNode>, Double> predLog = res.second;
		    CustomCluster clust = new CustomCluster(clf, dataMaker, dataWrapper, train);
		    ArrayList<HashSet<EventNode>> cdecChains = new ArrayList<HashSet<EventNode>>();

		    if(useGoldDocClusters) {
		    	HashMap<String, String> goldTestDocClusters = Utils.goldDocClusterTofName(testTopics, dataWrapper);
				for(String c_id : goldTestDocClusters.keySet())
					cdecChains.addAll(clust.cluster(predLog, predictionCutoff, goldTestDocClusters.get(c_id).split(" "), dataWrapper.missingTestEvs));
		    }
		    else {
		    	for(String c_id : testPredDocClusters.keySet())
					cdecChains.addAll(clust.cluster(predLog, predictionCutoff, testPredDocClusters.get(c_id).split(" "), dataWrapper.missingTestEvs));
		    }
		    	
			
			/*
			 * [4] Evaluate CoNLL performance
			 */
			
			/*
			 * Make CoNLL files
			 */
			List<File> testFiles = ECBWrapper.getFilesFromTopics(testTopics);
			File gold = ConllFile.makeConllFile(dataWrapper, dataWrapper.testCorefChains, testFiles, "gold.conll");
			File model = ConllFile.makeConllFile(dataWrapper, cdecChains, testFiles, "model.conll");
			
			/*
			 * Record performance
			 */
			PerformanceMetric conllRes = PerformanceMetric.getConllScores(gold, model, "CDEC");
			System.out.println(conllRes.getConllMetrics());
			for(String m : conllMetrics) {
				for(String t : types) {
					String name = m + "_" + t;
					scores.get(name).add(conllRes.getMetric(name));
				}
			}
			scores.get("conll_f1").add(conllRes.getMetric("conll_f1"));
		    
		} // CDEC done for this fold
		
		/*
		 * [5] Store performance results in .csv
		 */
		Utils.logResults(scores, logMessage);
	} // main
}