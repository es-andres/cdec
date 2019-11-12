package common;

import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;

import java.io.*;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.nio.file.Paths;
import java.sql.Timestamp;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

import org.spark_project.guava.io.Files;

import com.google.common.base.Charsets;
import com.google.common.base.Functions;
import com.google.common.base.Splitter;
import com.google.common.collect.Lists;
import com.google.common.math.Stats;

import clustering.EquivCluster;
import comparer.EvPairDataset;
import edu.stanford.nlp.util.StringUtils;
import tests.Experiments;
import naf.EventNode;


public class Main {
	
	static final Logger LOGGER = Logger.getLogger(Main.class.getName());
	/*
	 * ... seems have to do something a bit more nuanced than just highest precision
	 * Experiments:
	 * 
	 * 1. balanced train, filtered test, f1 optim. 
	 * 2. balanced train, filtered test, precision optim
	 * 3. balanced train, filtered test, f_beta=0.5 optim
	 * 4. balanced train, un-filtered test, f1 optim
	 * 5. balanced train, un-filtered test, f_beta=2 optim
	 * 6. un-balanced train, un-filtered test, f1 optim
	 * 7. actually balanced train, un-filtered test, f1 optim
	 * 
	 *  ... let's say 7 is best (< 4 by like .2, negligible)
	 * pick best performance, and then:
	 * 8. config 7, 5-fold cv
	 * 9. config 7, gold doc cluster 5-fold cv
	 * 10. config. 7, 36-45
	 * 11. config. 7, 36-45, gold doc clusters
	 * 12. config. 7, newsreader
	 * 13. config. 7, 2-fold
	 * 14. config. 7, shuffled 5-fold cv
	 * 15. config. 7, 4-fold cv no shuffle
	 * 16. config. 7, newsreader, trying to improve ceafe
	 * 17. config. 7 newsreader, trying to improve ceafe (cluster cutoff = 0.6)
	 * 18. newsreader, trying to improve ceafe (cluster cutoff q = 0.5)
	 * 
	 * 10. config 7, gold doc cluster, cv, // TODO: this != gold ev label distribution?
	 * 13. config. on a cv run with shuffled vs. unshuffled topics (are topics grouped contiguously by "topic"?)
	 */
//	"(4.%s) gold doc clusters, shuffle %s-cv, beta=1.6"
	//TODO: run best "balanced" config with actually balanced... false size was set to 3 times true size
	public static void main(String[] args) throws IOException
	{

		for(int k : Arrays.asList(5, 10)) {
			int offset = k == 5 ? 0 : 10;
			for(int r = 1; r < 11; r++) {
				int run = r + offset;
		
		double beta = 0.55;
		String logMessage = String.format("(%s-cv.%s) doc cluster model, balanced test, shuffle %s-cv, beta=%s", k, run, k,beta);
//		String logMessage = String.format("no reps 36-45; beta=%s", beta);
		String experiment_id = String.format("%s-cv.%s", k, run);
//		String experiment_id = "actually balanced train";
		boolean useGoldDocClusters = false;
		
//		SerUtils.cacheSemanticParses();
		
		/*
		 * [0] Prepare cross validation folds
		 */
		// make list of topics for CV fold generation
		List<Integer> topics = IntStream.rangeClosed(1,45).boxed().collect(Collectors.toList());
		topics.removeAll(Globals.DEAD_TOPICS); // some topics numbers don't exist in ECB+
		ECBWrapper dataWrapper = new ECBWrapper(topics); // loads all documents, including corenlp parses

//		int k = 2;
		boolean shuffle = true;
//		LinkedList<List<Integer>> kTestSets = new LinkedList<List<Integer>>();
		LinkedList<List<Integer>> kTestSets = Utils.cvFolds(topics, k, shuffle); // cv
//		kTestSets.add(IntStream.range(24, 44).boxed().collect(Collectors.toList())); // Newsreader
//		kTestSets.add(Arrays.asList(36,37,38,39,41,42,43,44,45)); // from papers
		
    	String[] conllMetrics = {"muc", "bcub", "ceafm", "ceafe", "blanc"};
    	String[] types = {"f1", "recall", "precision"};
		HashMap<String, ArrayList<Double>> scores = Utils.initScoreMap(conllMetrics, types); // performance scores stored here
		
		/*
		 * 0.1 Run CDEC procedure on each fold
		 */
		LOGGER.info(String.format("\nRunning %s-fold CV\n", k));
		int clustNum = 1;
		for (List<Integer> testTopics : kTestSets) {
			LOGGER.info(String.format("Fold %s -> Test set: %s", clustNum++, testTopics));
		    
			/*
		     * [1] Document clustering on test documents
		     */
			
		    /*
		     * Call python script to cluster documents
		     */
			String n_clusts = ""; // necessary if using k-means document clustering. 
			  					  // "" -> affinity propagationtestSets
			Process proc = new ProcessBuilder("bash", 
											  Globals.PY_DOC_CLUSTER.toString(), 
											  String.join(" ", Lists.transform(testTopics, Functions.toStringFunction())), 
											  n_clusts).start();
			BufferedReader output = new BufferedReader(new InputStreamReader(proc.getInputStream()));
			String[] s = output.readLine().split("BREAK");
			Map<String, String> testPredDocClusters = Splitter.on(",").withKeyValueSeparator(":").split(s[0].replace(" '", "").replace("'", ""));
			Map<String, String> docClusterPerf = Splitter.on(",").withKeyValueSeparator(":").split(s[1].replace(" '", "").replace("'",""));
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
		    List<List<EventNode>> trainPairs = dataWrapper.buildTrainPairs(trainTopics, balancedTrain, trainfNameToGoldDocClustMap);
		    
		    /*
		     * 2.2 Build event-pair test set
		     */
		    HashMap<String, String> testFNameToDocClustMap = null;
		    if(useGoldDocClusters)
		    	testFNameToDocClustMap = Utils.fNameToGoldDocCluster(testTopics, dataWrapper);
		    else
		    	testFNameToDocClustMap = Utils.fNameToPredictedDocCluster(testPredDocClusters); // predicted doc clusters
//			Utils.fNameToGoldDocCluster(testTopics, dataWrapper); // gold doc clusters
			// TODO: boolean to filter or unfilter test? or maybe just hard-code for experiments
			boolean filter = false;
			boolean balancedTest = true;
		    double evDistCutoff = 0;//Utils.computeTrainEvDistCutoff(dataWrapper);
		    double docSimCutoff = 1;//Utils.computeTrainDocSimCutoff(dataWrapper);
			List<List<EventNode>> testPairs = dataWrapper.buildTestPairs(testTopics, testFNameToDocClustMap, evDistCutoff, docSimCutoff, filter, balancedTest);
			
		    /*
		     * 2.3 log number of true/false in filtered dataset using doc clusters and gold dataset
		     */
//			dataWrapper.getGoldEvPairDistribution(testTopics) // gold using gold clusters
//			dataWrapper.buildTestPairs(testTopics, fNameToDocClustMap, evDistCutoff, docSimCutoff, false) // gold using model clusters
			// with doc clustering
		    HashMap<String, Double> labelDist = ECBWrapper.getLabelDistribution(dataWrapper.buildTestPairs(testTopics, testFNameToDocClustMap, evDistCutoff, docSimCutoff, false, balancedTest));
		    scores.get("unfilt_true_pairs").add(labelDist.get("true"));
		    scores.get("unfilt_false_pairs").add(labelDist.get("false"));
		    System.out.println("unfilt: " + labelDist);
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
		     */
			EvPairDataset dataMaker = new EvPairDataset();
		    Instances train = dataMaker.makeDataset(dataWrapper, trainPairs, Globals.LEMMATIZE, Globals.POS);
		    
		    /*
		     * 2.5 Train ev-pair classifier
		     */
		    MultilayerPerceptron clf = new MultilayerPerceptron();
		    clf.setLearningRate(0.001); 
//		    clf.setHiddenLayers();
		    clf.setMomentum(0.8);
		    
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
			
		    GeneralTuple<Double, HashMap<HashSet<EventNode>, Double>> res = Utils.testClassifier(clf, testPairs, dataMaker, 
		    																					 dataWrapper, train, scores,
		    																					 beta);
		    double predictionCutoff = res.first;
		    HashMap<HashSet<EventNode>, Double> predLog = res.second;
		    EquivCluster clust = new EquivCluster(clf, dataMaker, dataWrapper, train);
		    ArrayList<HashSet<EventNode>> cdecChains = new ArrayList<HashSet<EventNode>>();
		    //testPredDocClusters.keySet()
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
			PerformanceMetric conllRes = Experiments.getConllScores(gold, model, "CDEC");
			System.out.println(conllRes.getConllMetrics());
			for(String m : conllMetrics) {
				for(String t : types) {
					String name = m + "_" + t;
					scores.get(name).add(conllRes.getMetric(name));
				}
			}
			scores.get("conll_f1").add(conllRes.getMetric("conll_f1"));

//			if(clustNum == 3)
//				break;
		} // CDEC done for this fold
		
		/*
		 * [5] Store performance results in .csv
		 */
		Utils.logResults(scores, logMessage, experiment_id);
			} // cv rounds
		} // cv rounds

	    
	} // end CDEC
}