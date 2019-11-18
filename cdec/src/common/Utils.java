package common;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.sql.Timestamp;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;
import java.util.stream.IntStream;

import org.apache.commons.math3.ml.distance.DistanceMeasure;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.spark_project.guava.io.Files;

import com.google.common.base.Charsets;
import com.google.common.math.Quantiles;

import ecb_utils.ECBDoc;
import ecb_utils.ECBWrapper;
import ecb_utils.EventNode;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.pipeline.CoreSentence;
import edu.stanford.nlp.util.StringUtils;
import feature_extraction.EvPairDataset;
import feature_extraction.TFIDF;
import me.tongfei.progressbar.ProgressBar;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Misc. procedures that could be in main but add too much visual clutter
 * @author acrem003
 *
 */
public class Utils {
	
	private static final Logger LOGGER = Logger.getLogger(Utils.class.getName());
	
	
	public static void logResults(HashMap<String, ArrayList<Double>> scores, String logMessage, String experiment_id) {
		String time = new Timestamp(System.currentTimeMillis()).toString().replace(" ", "_");

		/*
		 * all scores
		 */
		List<String> scoreList = new LinkedList<String>(scores.keySet());
		scoreList.add(0, "experiment_id");
		StringBuilder csvAll = new StringBuilder(String.join(",", scoreList) + "\n");
		for(int i = 0; i < scores.get(scoreList.get(1)).size(); i++) {
			String row = experiment_id + ",";
			for(String score : scoreList) {
//				System.out.println(score);
				if(!score.equals("experiment_id")) {
					row += scores.get(score).get(i) + ",";					
				}
			}
			row = row.substring(0, row.length() - 1) + "\n";
			
			csvAll.append(row);
		}
		
		/*
		 * avg scores
		 */
		List<String> toAvg = Arrays.asList("experiment_id", "conll_f1", "clf_f1", "clf_precision", "clf_recall");
		StringBuilder csvAvg = new StringBuilder(String.join(",", toAvg) +  "\n");
		String row = "";
		for(String score : toAvg) {
			if(score.equals("experiment_id"))
				row = experiment_id + ",";
			else
				row += scores.get(score).stream().mapToDouble(a -> a).average().getAsDouble() +  ",";
		}
		row = row.substring(0, row.length() - 1);
		csvAvg.append(row);
		
		/*
		 * log
		 */
		List<String> logCols = Arrays.asList("experiment_id", "file_name", "description", "conll_f1",
												"muc_f1", "bcub_f1", "ceafe_f1",
												"clf_f1", "clf_precision", "clf_recall", "clf_accuracy");
		File logCsvFile = Paths.get(Globals.RESULTS_DIR.toString(),  "log.csv").toFile();
		if(!logCsvFile.exists()) {
			try {
				logCsvFile.createNewFile();
				StringBuilder logCsvHeader = new StringBuilder(String.join(",", logCols) + "\n");
				Files.append(logCsvHeader.toString(), logCsvFile, Charsets.UTF_8);
			}
			catch(IOException e) {
				e.printStackTrace();
			}
			
		}
		String logRow = "";
		for(String colName : logCols) {
			if(colName.equals("experiment_id"))
				logRow = experiment_id + ",";
			else if(colName.equals("file_name"))
				logRow += time + ",";
			else if(colName.equals("description"))
				logRow += logMessage.replace(",",";") + ",";
			else
				logRow += scores.get(colName).get(scores.get(colName).size() - 1) +  ",";
//				logRow += scores.get(colName).stream().mapToDouble(a -> a).average().getAsDouble() + ",";
		}
		logRow = logRow.substring(0, logRow.length() - 1) + "\n";
		
		File allFile = Paths.get(Globals.RESULTS_DIR.toString(), time + ".csv").toFile();
		File avgFile = Paths.get(Globals.RESULTS_DIR.toString(), time + "_avg.csv").toFile();
		File logTxtFile = Paths.get(Globals.RESULTS_DIR.toString(), "log.txt").toFile();
		
		try {
			// scores
			Files.write(csvAll, allFile, Charsets.UTF_8);
			Files.write(csvAvg, avgFile, Charsets.UTF_8);
			// log
			logTxtFile.createNewFile();
			logMessage = time + ": " + logMessage + "\n" + StringUtils.repeat("-", 8) + "\n";
			Files.append(logMessage, logTxtFile, Charsets.UTF_8);
			Files.append(logRow, logCsvFile, Charsets.UTF_8);
		} catch (IOException e) {
			e.printStackTrace();
		}

		LOGGER.info("Results written to " + allFile);
		LOGGER.info("Avgs. written to " + avgFile);
		LOGGER.info("Log written to " + logTxtFile);
	}
    /**
     * Constructor.
     * @param n the number of samples.
     * @param k the number of rounds of cross validation.
     * @param permutate determiner of index permutation
     */
    public static LinkedList<List<Integer>> cvFolds(List<Integer> topics, int k, boolean permutate) {
    	int n = topics.size();
        if (n < 0) {
            throw new IllegalArgumentException("Invalid sample size: " + n);
        }

        if (k < 0 || k > n) {
            throw new IllegalArgumentException("Invalid number of CV rounds: " + k);
        }


        int[] index;
        if (permutate){
            index = smile.math.Math.permutate(n);
        }
        else{
            index = new int[n];
            for (int i = 0; i < n; i++) {
                index[i] = i;
            }
        }

        int[][] train = new int[k][];
        int[][] test = new int[k][];

        int chunk = n / k;
        for (int i = 0; i < k; i++) {
            int start = chunk * i;
            int end = chunk * (i + 1);
            if (i == k-1) end = n;

            train[i] = new int[n - end + start];
            test[i] = new int[end - start];
            for (int j = 0, p = 0, q = 0; j < n; j++) {
                if (j >= start && j < end) {
                    test[i][p++] = index[j];
                } else {
                    train[i][q++] = index[j];
                }
            }
        }
		LinkedList<List<Integer>> kTestSets = new LinkedList<List<Integer>>();
		for(int[] fold : test) {
			List<Integer> l = new LinkedList<Integer>();
			for(int idx : fold) {
				l.add(topics.get(idx));
			}
			kTestSets.add(l);
		}
        return kTestSets;
    }
	public static HashMap<String, ArrayList<Double>> initScoreMap(String[] conllMetrics, String[] types) {
		HashMap<String, ArrayList<Double>> scores = new HashMap<String, ArrayList<Double>>();
		new HashMap<String, ArrayList<Double>>(); // store performance metrics
    	for(String m : conllMetrics) {
    		for(String t : types)
    			scores.put(m  + "_" + t, new ArrayList<Double>());
    	}
    	
    	scores.put("conll_f1", new ArrayList<Double>());
    	for(String t : types)
    		scores.put("clf_"+ t, new ArrayList<Double>());
    	scores.put("clf_cutoff", new ArrayList<Double>());
    	scores.put("clf_accuracy", new ArrayList<Double>());
    	
    	for(String t : Arrays.asList("train", "this", "unfilt", "gold"))
    		for(String b : Arrays.asList("true", "false"))
    			scores.put(t+"_"+b+"_pairs", new ArrayList<Double>());
    	
    	for(String t : Arrays.asList("v-measure", "completeness", "homogeneity", "ari"))
    		scores.put(t, new ArrayList<Double>());

    	return scores;
	}
	
	public static HashMap<String, String> fNameToPredictedDocCluster(Map<String, String> map){
		HashMap<String, String> invertedMap = new HashMap<String, String>();
		for(String clustId : map.keySet()) {
			for(String fName : map.get(clustId).split(" ")) {
				invertedMap.put(fName, clustId);
			}
		}
		return invertedMap;
	}
	
	public static HashMap<String, String> fNameToGoldDocCluster(List<Integer> testTopics, ECBWrapper dataWrapper){
		HashMap<String, String> map = new HashMap<String, String>();
		List<File> files = ECBWrapper.getFilesFromTopics(testTopics);
		for(File f : files)
			map.put(f.getName(), dataWrapper.docs.get(f.getName()).getTopic() + "_" + dataWrapper.docs.get(f.getName()).getSubTopic());
			
		return map;	
	}
	
	public static HashMap<String, String> goldDocClusterTofName(List<Integer> topics, ECBWrapper dataWrapper){
		HashMap<String, LinkedList<String>> map = new HashMap<String, LinkedList<String>>();
		List<File> files = ECBWrapper.getFilesFromTopics(topics);
		for(File f : files) {
			ECBDoc doc = dataWrapper.docs.get(f.getName());
			String topicKey = doc.getTopic() + "_" + doc.getSubTopic();
			if(!map.containsKey(topicKey))
				map.put(topicKey, new LinkedList<String>());
			map.get(topicKey).add(f.getName());
		}
		HashMap<String, String> strMap = new HashMap<String, String>();
		for(String key : map.keySet())
			strMap.put(key, String.join(" ", map.get(key)));
			
		return strMap;	
	}
	
	public static double computeTrainEvDistCutoff(ECBWrapper dataWrapper) {
		
		HashSet<HashSet<EventNode>> seen = new HashSet<HashSet<EventNode>>();
		SummaryStatistics trueDists = new SummaryStatistics();
		SummaryStatistics falseDists = new SummaryStatistics();
		for(EventNode ev1 : ProgressBar.wrap(dataWrapper.trainCorefGraph.nodes(),"Computing evSim cutoff")) {
			for(EventNode ev2 : dataWrapper.trainCorefGraph.nodes()) {
//				double docSim = Transforms.cosineSim(dataWrapper.docs.get(ev1.file.getName()).tfidfVec, 
//										   dataWrapper.docs.get(ev2.file.getName()).tfidfVec);
				if(!ev1.equals(ev2) && ev1.getTopic().equals(ev2.getTopic()) 
						&& ev1.getSubTopic().equals(ev2.getSubTopic())
						&& !ev1.file.equals(ev2.file) // cross doc
						&& seen.add(new HashSet<EventNode>(Arrays.asList(ev1,ev2)))) {
					
					List<IndexedWord> ev1Text = EvPairDataset.mainEvText(ev1, dataWrapper.docs.get(ev1.file.getName()));
					List<IndexedWord> ev2Text = EvPairDataset.mainEvText(ev2, dataWrapper.docs.get(ev2.file.getName()));
					List<CoreSentence> ev1Sentences = Arrays.asList(EvPairDataset.evMainSentence(ev1, dataWrapper.docs.get(ev1.file.getName())));
					List<CoreSentence> ev2Sentences = Arrays.asList(EvPairDataset.evMainSentence(ev2, dataWrapper.docs.get(ev2.file.getName())));
					List<CoreSentence> sentenceCorpus = EvPairDataset.makeSentCorpus(ev1Sentences, ev2Sentences, ev1.file.equals(ev2.file));
					TFIDF comparer = new TFIDF(sentenceCorpus, Globals.LEMMATIZE, Globals.POS, 1);
					DistanceMeasure dist = new EuclideanDistance();
					double evDist = dist.compute(comparer.makeEvVector(ev1Text, ev1Sentences), comparer.makeEvVector(ev2Text, ev2Sentences));
					if(ev1.corefers(ev2))
						trueDists.addValue(evDist);
					else
						falseDists.addValue(evDist);
				}
			}
		}
//		System.out.println("true: " + trueDists.getMean() + " +/- " + trueDists.getVariance());
//		System.out.println("false: " + falseDists.getMean() + " +/- " + falseDists.getVariance() + "\n");
		return falseDists.getMean();
	}
	
	public static double computeTrainDocSimCutoff(ECBWrapper dataWrapper) {
		HashSet<HashSet<File>> seen = new HashSet<HashSet<File>>();
		SummaryStatistics trueSims = new SummaryStatistics();
		SummaryStatistics falseSims = new SummaryStatistics();
		List<File> trainFiles = ECBWrapper.getFilesFromTopics(dataWrapper.trainTopics);
		
		for(File f1 : ProgressBar.wrap(trainFiles,"Computing doc dist cutoff")) {
			for(File f2 : trainFiles) {
				ECBDoc doc1 = dataWrapper.docs.get(f1.getName());
				ECBDoc doc2 = dataWrapper.docs.get(f2.getName());
				if(!f1.equals(f2) && doc1.getTopic().equals(doc2.getTopic()) 
						&& doc1.getSubTopic().equals(doc2.getSubTopic())
						&& seen.add(new HashSet<File>(Arrays.asList(f1,f2)))) {
					
					double docSim = Transforms.cosineSim(doc1.tfidfVec, doc2.tfidfVec);
					if(ECBWrapper.coferer(doc1, doc2))
						trueSims.addValue(docSim);
					else
						falseSims.addValue(docSim);
				}
			}
		}
//		System.out.println("true: " + trueSims.getMean() + " +/- " + trueSims.getVariance());
//		System.out.println("false: " + falseSims.getMean() + " +/- " + falseSims.getVariance() + "\n");
		return falseSims.getMean();
	}
	
	
	public static double computeTestDocSimCutoff() {
		
		return 0;
	}
	
	public static GeneralTuple<Double, HashMap<HashSet<EventNode>, Double>> testClassifier(Classifier clf, LinkedList<GeneralTuple<Instance, List<EventNode>>> test, 
																	 Instances train, HashMap<String, ArrayList<Double>> scores,
																	 double beta) {
		LOGGER.info("Testing classifier and finding optimal prediction cutoff");
//		double tp = 0;
//		double fp = 0;
//		double fn = 0;
//		double tn = 0;
		ArrayList<Double> preds = new ArrayList<Double>();
		ArrayList<Integer> labels = new ArrayList<Integer>();
		HashMap<HashSet<EventNode>, Double> predLog = new HashMap<HashSet<EventNode>, Double>();
//		for(List<EventNode> pair : testPairs) {
//			Instance inst = dataMaker.evPairVector(dataWrapper, pair, Globals.LEMMATIZE, Globals.POS);
//			inst.setDataset(train);
//			double pred;
//			try {
//				pred = clf.distributionForInstance(inst)[1];
//				preds.add(pred);
//				labels.add((int)inst.classValue());
//				predLog.put(new HashSet<EventNode>(pair), pred);
//			} catch (Exception e) {
//				e.printStackTrace();
//			}
//
//		}
		for(GeneralTuple<Instance, List<EventNode>> tup : test) {
			Instance inst = tup.first;
			List<EventNode> pair = tup.second;
			double pred;
			try {
				pred = clf.distributionForInstance(inst)[1];
				preds.add(pred);
				labels.add((int)inst.classValue());
				predLog.put(new HashSet<EventNode>(pair), pred);
			} catch (Exception e) {
				e.printStackTrace();
			}

		}

		double max_cutoff = Double.MIN_VALUE;
		double max_recall = Double.MIN_VALUE;
		double max_precision = Double.MIN_VALUE;
		double max_f1 = Double.MIN_VALUE;
		double max_fB = Double.MIN_VALUE;
		double max_accuracy = Double.MIN_VALUE;
		for(int quant : IntStream.range(60, 99).toArray()) {
			double tp = 0;
			double fp = 0;
			double fn = 0;
			double tn = 0;
			double cutoff = Quantiles.percentiles().index(quant).compute(preds);
			for(int i = 0; i < preds.size(); i++) {
				double conf = preds.get(i);
				int label = labels.get(i);
				double pred = (conf >= cutoff) ? 1 : 0;
				boolean correct = (label == pred);
				if(correct && pred == 1)
					tp++;
				else if(correct && pred == 0)
					tn++;
				else if(!correct && pred == 1)
					fp++;
				else if(!correct && pred == 0)
					fn++;
			}
			double accuracy = (tp + tn) / (tp + tn + fp + fn);
			double recall = tp / (tp + fn);
			double precision = tp / (tp + fp);
			double f1 = 2*((precision*recall)/(precision+recall));
			double fB = (1 + Math.pow(beta, 2))*((precision*recall)/(beta*precision+recall));
			if(fB > max_fB) {
				max_fB = fB;
				max_recall = recall;
				max_precision = precision;
				max_f1 = f1;
				max_cutoff = cutoff;
				max_accuracy = accuracy;
			}
		}
		System.out.println("recall: " + max_recall);
		System.out.println("precision: " + max_precision);
		System.out.println("accuracy: " + max_accuracy);
		System.out.println("f1: " + max_f1);

	    scores.get("clf_f1").add(max_f1);
	    scores.get("clf_recall").add(max_recall);
	    scores.get("clf_precision").add(max_precision);
	    scores.get("clf_accuracy").add(max_accuracy);
	    scores.get("clf_cutoff").add(max_cutoff);
	    
	    return new GeneralTuple<Double, HashMap<HashSet<EventNode>, Double>>(max_cutoff, predLog);
		
	}
}
