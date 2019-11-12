package tests;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.util.List;

import common.ArffWriter;
import common.ConllFile;
import common.Globals;
import common.PerformanceMetric;
import coref.CDEC;
import coref.WDEC;
import weka.Weka;

public class Experiments {

	public Experiments() {}
	
	public static PerformanceMetric getConllScores(File goldConll, File modelConll, String name) {
		
		/*
		 * run conll scorer
		 */
		StringBuilder builder = new StringBuilder();
		try {
			Process process = new ProcessBuilder("perl", Globals.CONLL_SCORER_PATH.toString(), "all", 
												 goldConll.toString(), modelConll.toString()).start();
			BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
			String line = null;
			while ( (line = reader.readLine()) != null) {
			   builder.append(line);
			   builder.append(System.getProperty("line.separator"));
			}
		}
		catch(Exception e) {
			e.printStackTrace();
		}
		PerformanceMetric metrics = new PerformanceMetric();
		String result = builder.toString();

		metrics.addConllMetrics(result, false);

		return metrics;
	}
	
	/*
	 * Runs WDEC using gold event labels
	 */
	public WDEC doGoldWDEC(List<File> trainFiles, List<File> testFiles, PerformanceMetric wdecPerformanceMetrics, File writeDir, String classifierType, boolean useTime) {

		//########################### Classifier #################################
		
		/*
		 * make feature file for pairwise classifier
		 */
		ArffWriter arff = new ArffWriter();
		File featureFile = null;
//		featureFile = arff.writeWDECArffFromGold(trainFiles, "10fold", writeDir.toString(),"WDEC");
		/*
		 * build classifier
		 */
		Weka wdEvClassifier = null;//new Weka(false);
//		wdEvClassifier.trainClassifier(featureFile.toString(),"WDEC", writeDir.toString(), classifierType);

		//########################### WDEC #################################
		
		/*
		 * WDEC
		 */
		WDEC wdec = null;
//		WDEC wdec = new WDEC(testFiles,wdEvClassifier);
		boolean transitive = true;
		boolean crossDoc = true;
		wdec.doGoldWDEC(transitive, crossDoc);
		
		//########################### Scorer #################################
		
		/*
		 * make conll files
		 */
		ConllFile conll = new ConllFile();
//		File goldConll = conll.writeWDConllFromGold(testFiles, writeDir.toString());
//		File modelConll = conll.writeWDConllFromPreds(wdec, writeDir.toString());
		File goldConll = null;
		File modelConll = null;
		
		/*
		 * run conll scorer
		 */
		String cmd = Globals.CONLL_SCORER_PATH + "/" + "perl_scorer";
		StringBuilder builder = new StringBuilder();
		try {
			Process process = new ProcessBuilder("sh",cmd,goldConll.toString(),modelConll.toString()).start();
			BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
			String line = null;
			while ( (line = reader.readLine()) != null) {
			   builder.append(line);
			   builder.append(System.getProperty("line.separator"));
			}
		}
		catch(Exception e) {
			e.printStackTrace();
		}
		
		String result = builder.toString();
		wdecPerformanceMetrics.addConllMetrics(result,true);
		System.out.println("Cumulative (ie. averaged) WDEC performance scores so far:");
		System.out.println("\n" + wdecPerformanceMetrics.getConllMetrics() + "\n");
		
		return wdec;
		
	}
	
	/*
	 * Runs CDEC on output of WDEC
	 */
	public void doGoldCDEC(WDEC wdecResults, PerformanceMetric cdecPerformanceMetrics, File writeDir, String classifierType,boolean useTime) {
		
		//########################### Classifier #################################
		
		/*
		 * make feature file for pairwise classifier
		 */
		ArffWriter arff = new ArffWriter();
		File featureFile = null;
//		featureFile= arff.writeCDECArffFromGold(wdecResults.trainFiles, wdecResults.fileToM_idToWD_id, writeDir.toString(),useTime);
		
		/*
		 * build classifier
		 */
		Weka cdecClassifier = null;//new Weka(false);
//		cdecClassifier.trainClassifier(featureFile.toString(),"CDEC", writeDir.toString(), classifierType);

		//########################### CDEC #################################
		
		/*
		 * CDEC
		 */

		CDEC cdec = null;
		boolean transitive = true;
//		cdec.doGoldCDEC(transitive, useTime);
		
		//########################### Scorer #################################
		
		/*
		 * make conll files
		 */
		ConllFile conll = new ConllFile();
		File goldConll = null;
		File modelConll = null;
		
		/*
		 * run conll scorer
		 */
		String cmd = Globals.CONLL_SCORER_PATH + "/" + "perl_scorer";
		StringBuilder builder = new StringBuilder();
		try {
			Process process = new ProcessBuilder("sh",cmd,goldConll.toString(),modelConll.toString()).start();
			BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
			String line = null;
			while ( (line = reader.readLine()) != null) {
			   builder.append(line);
			   builder.append(System.getProperty("line.separator"));
			}
		}
		catch(Exception e) {
			e.printStackTrace();
		}

		String result = builder.toString();
		cdecPerformanceMetrics.addConllMetrics(result,true);
		System.out.println("Cumulative (ie. averaged) CDEC performance scores so far:");
		System.out.println("\n" + cdecPerformanceMetrics.getConllMetrics() + "\n");

	}
	
}
