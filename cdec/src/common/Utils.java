package common;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
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


import com.google.common.base.Charsets;
import com.google.common.base.Functions;
import com.google.common.collect.Lists;
import com.google.common.io.FileWriteMode;
import com.google.common.io.Files;
import com.google.common.math.Quantiles;

import ecb_utils.ECBDoc;
import ecb_utils.ECBWrapper;
import ecb_utils.EventNode;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVSaver;

/**
 * Misc. procedures that could be in main but add too much visual clutter
 * @author acrem003
 *
 */
public class Utils {
	
	private static final Logger LOGGER = Logger.getLogger(Utils.class.getName());
	
	public static HashMap<String, HashSet<String>> makeCleanSentenceDict(File cleanTable) {
		HashMap<String, HashSet<String>> records = new HashMap<String,HashSet<String>>();

		try (BufferedReader br = new BufferedReader(new FileReader(cleanTable))) {
			String line;
			while ((line = br.readLine()) != null) {
				String[] values = line.split(",");
				String fileName = values[0] + "_" + values[1];
				String sentenceNum = values[2].replaceAll(" ", "");
				if(!records.containsKey(fileName))
					records.put(fileName, new HashSet<String>());
				records.get(fileName).add(sentenceNum);
			}
		}
		catch(IOException e) {
			e.printStackTrace();
		}

		return records;
	}
	
	public static void saveCsv(Instances train) {
	    CSVSaver csv = new CSVSaver();
	    try {
			Paths.get(Globals.ROOT_DIR.toString(), "data", "train.csv").toFile().createNewFile();
		    csv.setFile(Paths.get(Globals.ROOT_DIR.toString(), "data", "train.csv").toFile());
		    csv.setInstances(train);
		    csv.setDestination(Paths.get(Globals.ROOT_DIR.toString(), "data", "train.csv").toFile());
		    csv.writeBatch();
		} catch (IOException e) {
			e.printStackTrace();
		}

	}
	public static void logResults(HashMap<String, ArrayList<Double>> scores, String logMessage) {
		String time = new Timestamp(System.currentTimeMillis()).toString().replace(" ", "_");

		/*
		 * all scores
		 */
		List<String> scoreList = new LinkedList<String>(scores.keySet());
		scoreList.add(0, "experiment_id");
		StringBuilder csvAll = new StringBuilder(String.join(",", scoreList) + "\n");
		for(int i = 0; i < scores.get(scoreList.get(1)).size(); i++) {
			String row = logMessage + ",";
			for(String score : scoreList) {
				if(!score.equals("experiment_id")) {
					row += scores.get(score).get(i) + ",";					
				}
			}
			row = row.substring(0, row.length() - 1) + "\n";
			
			csvAll.append(row);
		}
		
		
		/*
		 * log
		 */
		List<String> logCols = Arrays.asList("experiment_id", "file_name", "description", "conll_f1",
												"muc_f1", "bcub_f1", "ceafe_f1",
												"clf_f1", "clf_precision", "clf_recall", "clf_accuracy", "beta");
//		List<String> logCols = Arrays.asList("experiment_id", "file_name", "description", "conll_f1", "beta");
		File logCsvFile = Paths.get(Globals.RESULTS_DIR.toString(),  "log.csv").toFile();
		if(!logCsvFile.exists()) {
			try {
				logCsvFile.createNewFile();
				StringBuilder logCsvHeader = new StringBuilder(String.join(",", logCols) + "\n");
				Files.asCharSink(logCsvFile, Charsets.UTF_8, FileWriteMode.APPEND).write(logCsvHeader.toString());
			}
			catch(IOException e) {
				e.printStackTrace();
			}
			
		}
		String logRow = "";
		for(String colName : logCols) {
			if(colName.equals("experiment_id"))
				logRow = logMessage+ ",";
			else if(colName.equals("file_name"))
				logRow += time + ",";
			else if(colName.equals("description"))
				logRow += logMessage.replace(",",";") + ",";
			else {
				logRow += scores.get(colName).get(scores.get(colName).size() - 1) +  ",";
			}
//				logRow += scores.get(colName).stream().mapToDouble(a -> a).average().getAsDouble() + ",";
		}
		logRow = logRow.substring(0, logRow.length() - 1) + "\n";
		
		File allFile = Paths.get(Globals.RESULTS_DIR.toString(), time + ".csv").toFile();
		File avgFile = Paths.get(Globals.RESULTS_DIR.toString(), time + "_avg.csv").toFile();
		File logTxtFile = Paths.get(Globals.RESULTS_DIR.toString(), "log.txt").toFile();
		
		try {
			// scores
			Files.asCharSink(allFile, Charsets.UTF_8).write(csvAll);
			// log
			Files.asCharSink(logCsvFile, Charsets.UTF_8, FileWriteMode.APPEND).write(logRow);
		} catch (IOException e) {
			e.printStackTrace();
		}

		LOGGER.info("Results written to " + allFile);
		LOGGER.info("Avgs. written to " + avgFile);
		LOGGER.info("Log written to " + logTxtFile);
	}
	
    /**
     * Make cross validation folds using indexing on an array. 
     * 
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
    
    /**
     * Scores are stored here. 
     * 
     * @param conllMetrics
     * @param types
     * @return
     */
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
    	
    	for(String t : Arrays.asList("train", "this", "gold"))
    		for(String b : Arrays.asList("true", "false"))
    			scores.put(t+"_"+b+"_pairs", new ArrayList<Double>());
    	
    	for(String t : Arrays.asList("v-measure", "completeness", "homogeneity", "ari"))
    		scores.put(t, new ArrayList<Double>());
    	
    	scores.put("beta", new ArrayList<Double>());

    	return scores;
	}
	
	/**
	 * Makes a map from file name to the predicted document cluster
	 * @param map
	 * @return
	 */
	public static HashMap<String, String> fNameToPredictedDocCluster(Map<String, String> map){
		HashMap<String, String> invertedMap = new HashMap<String, String>();
		for(String clustId : map.keySet()) {
			for(String fName : map.get(clustId).split(" ")) {
				invertedMap.put(fName, clustId);
			}
		}
		return invertedMap;
	} 
	
	/**
	 * Makes a map from file name to gold-standard document cluster
	 * @param testTopics
	 * @param dataWrapper
	 * @return
	 */
	public static HashMap<String, String> fNameToGoldDocCluster(List<Integer> testTopics, ECBWrapper dataWrapper){
		HashMap<String, String> map = new HashMap<String, String>();
		List<File> files = ECBWrapper.getFilesFromTopics(testTopics);
		for(File f : files)
			map.put(f.getName(), dataWrapper.docs.get(f.getName()).getTopic() + "_" + dataWrapper.docs.get(f.getName()).getSubTopic());
			
		return map;	
	}
	
	/**
	 * Makes a map from gold-standard document cluster to a list of its files
	 * @param topics
	 * @param dataWrapper
	 * @return
	 */
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
	
	
	public static GeneralTuple<Double, HashMap<HashSet<EventNode>, Double>> testClassifier(Classifier clf, 
																	 LinkedList<GeneralTuple<Instance, List<EventNode>>> test,
																	 LinkedList<GeneralTuple<Instance, List<EventNode>>> dev, 
																	 HashMap<String, ArrayList<Double>> scores,
																	 double beta) {
		LOGGER.info("Testing classifier and finding optimal prediction cutoff");

		HashMap<HashSet<EventNode>, Double> testPredLog = new HashMap<HashSet<EventNode>, Double>();
		for(GeneralTuple<Instance, List<EventNode>> tup : test) {
			Instance inst = tup.first;
			List<EventNode> pair = tup.second;
			double pred;
			try {
				pred = clf.distributionForInstance(inst)[1];
				testPredLog.put(new HashSet<EventNode>(pair), pred);
			} catch (Exception e) {
				e.printStackTrace();
			}

		}
		
		ArrayList<Double> devPreds = new ArrayList<Double>();
		ArrayList<Integer> devLabels = new ArrayList<Integer>();
		HashMap<HashSet<EventNode>, Double> devPredLog = new HashMap<HashSet<EventNode>, Double>();
		for(GeneralTuple<Instance, List<EventNode>> tup : dev) {
			Instance inst = tup.first;
			List<EventNode> pair = tup.second;
			double pred;
			try {
				pred = clf.distributionForInstance(inst)[1];
				devPreds.add(pred);
				devLabels.add((int)inst.classValue());
				devPredLog.put(new HashSet<EventNode>(pair), pred);
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
			double cutoff = Quantiles.percentiles().index(quant).compute(devPreds);
			for(int i = 0; i < devPreds.size(); i++) {
				double conf = devPreds.get(i);
				int label = devLabels.get(i);
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
		
		// COMMENTED FOR BETA
	    scores.get("clf_f1").add(max_f1);
	    scores.get("clf_recall").add(max_recall);
	    scores.get("clf_precision").add(max_precision);
	    scores.get("clf_accuracy").add(max_accuracy);
	    scores.get("clf_cutoff").add(max_cutoff);
	    
	    return new GeneralTuple<Double, HashMap<HashSet<EventNode>, Double>>(max_cutoff, testPredLog);
		
	}
}
