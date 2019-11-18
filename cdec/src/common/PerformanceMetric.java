package common;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.StringJoiner;

public class PerformanceMetric {

	private double tp;
	private double tn;
	private double fn;
	private double fp;
	private ArrayList<Double> mucF1;
	private ArrayList<Double> mucR;
	private ArrayList<Double> mucP;
	private ArrayList<Double> bcubF1;
	private ArrayList<Double> bcubR;
	private ArrayList<Double> bcubP;
	private ArrayList<Double> ceafmF1;
	private ArrayList<Double> ceafmR;
	private ArrayList<Double> ceafmP;
	private ArrayList<Double> ceafeF1;
	private ArrayList<Double> ceafeR;
	private ArrayList<Double> ceafeP;
	private ArrayList<Double> blancF1;
	private ArrayList<Double> blancR;
	private ArrayList<Double> blancP;
	private double conllF1;
	
	public PerformanceMetric() {
		tp = 0;
		tn = 0;
		fn = 0;
		fp = 0;
		mucF1 = new ArrayList<Double>();
		mucR = new ArrayList<Double>();
		mucP = new ArrayList<Double>();
		bcubF1 = new ArrayList<Double>();
		bcubR = new ArrayList<Double>();
		bcubP = new ArrayList<Double>();
		ceafmF1 = new ArrayList<Double>();
		ceafmR = new ArrayList<Double>();
		ceafmP = new ArrayList<Double>();
		ceafeF1 = new ArrayList<Double>();
		ceafeR = new ArrayList<Double>();
		ceafeP = new ArrayList<Double>();
		blancF1 = new ArrayList<Double>();
		blancR = new ArrayList<Double>();
		blancP = new ArrayList<Double>();
		conllF1 = 0;
	}
	
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
	
	public double getMetric(String s) {
		double res = -1;
		switch(s) {
		// muc
		case "muc_f1":
			res = mucF1.get(0);
			break;
		case "muc_recall":
			res = mucR.get(0);
			break;
		case "muc_precision":
			res = mucP.get(0);
			break;
		// bcub
		case "bcub_f1":
			res = bcubF1.get(0);
			break;
		case "bcub_recall":
			res = bcubR.get(0);
			break;
		case "bcub_precision":
			res = bcubP.get(0);
			break;
		// ceafm
		case "ceafm_f1":
			res = ceafmF1.get(0);
			break;
		case "ceafm_recall":
			res = ceafmR.get(0);
			break;
		case "ceafm_precision":
			res = ceafmP.get(0);
			break;
		// ceafe
		case "ceafe_f1":
			res = ceafeF1.get(0);
			break;
		case "ceafe_recall":
			res = ceafeR.get(0);
			break;
		case "ceafe_precision":
			res = ceafeP.get(0);
			break;
		// blanc
		case "blanc_f1":
			res = blancF1.get(0);
			break;
		case "blanc_recall":
			res = blancR.get(0);
			break;
		case "blanc_precision":
			res = blancP.get(0);
			break;
		// conll f1
		case "conll_f1":
			res = conllF1;
			break;
		}
		if(res == -1) {
			System.exit(-1);
		}
		return res;
	} 
	public void addConllMetrics(String result,boolean verbose) {
		boolean muc,bcub,ceafm,ceafe,blanc;
		muc = bcub = ceafm = ceafe = blanc= false;
		for(String line : result.split("\n")) {
			if(line.toLowerCase().contains("muc"))
				muc = true;
			if(line.toLowerCase().contains("bcub"))
				bcub = true;
			if(line.toLowerCase().contains("ceafm"))
				ceafm = true;
			if(line.toLowerCase().contains("ceafe"))
				ceafe = true;
			if(line.toLowerCase().contains("blanc") && !line.toLowerCase().contains("recall"))
				blanc = true;
			if(((line.toLowerCase().contains("coreference") && (line.toLowerCase().contains("recall"))) 
					|| line.toLowerCase().contains("blanc: recall")) 
					&& !line.toLowerCase().contains("links")) {

				if(muc) {
					String[] wholeLine = line.split("\\)");
					this.mucR.add(Double.parseDouble(wholeLine[1].split("%")[0].replace(" ", "")));
					this.mucP.add(Double.parseDouble(wholeLine[2].split("%")[0].replace(" ", "")));
					
					String f1 = line.substring(line.lastIndexOf(':') + 2);
					f1 = f1.substring(0,f1.length()-1);
					this.mucF1.add(Double.parseDouble(f1));
					muc = false;
				}
				if(bcub) {
					String[] wholeLine = line.split("\\)");
					this.bcubR.add(Double.parseDouble(wholeLine[1].split("%")[0].replace(" ", "")));
					this.bcubP.add(Double.parseDouble(wholeLine[2].split("%")[0].replace(" ", "")));
					String res = line.substring(line.lastIndexOf(':') + 2);
					res = res.substring(0,res.length()-1);
					this.bcubF1.add(Double.parseDouble(res));
					bcub = false;
				}
				if(ceafm) {
					String[] wholeLine = line.split("\\)");
					this.ceafmR.add(Double.parseDouble(wholeLine[1].split("%")[0].replace(" ", "")));
					this.ceafmP.add(Double.parseDouble(wholeLine[2].split("%")[0].replace(" ", "")));
					String res = line.substring(line.lastIndexOf(':') + 2);
					res = res.substring(0,res.length()-1);
					this.ceafmF1.add(Double.parseDouble(res));
					ceafm = false;
				}
				if(ceafe) {
					String[] wholeLine = line.split("\\)");
					this.ceafeR.add(Double.parseDouble(wholeLine[1].split("%")[0].replace(" ", "")));
					this.ceafeP.add(Double.parseDouble(wholeLine[2].split("%")[0].replace(" ", "")));
					String res = line.substring(line.lastIndexOf(':') + 2);
					res = res.substring(0,res.length()-1);
					this.ceafeF1.add(Double.parseDouble(res));
					ceafe = false;
				}
				if(blanc) {
					String[] wholeLine = line.split("\\)");
					this.blancR.add(Double.parseDouble(wholeLine[1].split("%")[0].replace(" ", "")));
					this.blancP.add(Double.parseDouble(wholeLine[2].split("%")[0].replace(" ", "")));
					String res = line.substring(line.lastIndexOf(':') + 2);
					res = res.substring(0,res.length()-1);
					this.blancF1.add(Double.parseDouble(res));
					blanc = false;
				}	
			}
		}
		ArrayList<Double> tempConll = new ArrayList<Double>();
		tempConll.addAll(this.mucF1);
		tempConll.addAll(this.bcubF1);
		tempConll.addAll(this.ceafeF1);
		conllF1 = tempConll.stream().mapToDouble(val -> val).average().orElse(0.0);
	}
	
	
	public String getConllMetrics() {
		StringJoiner res = new StringJoiner("\n");

		res.add("muc (recall): " + this.mucR.stream().mapToDouble(val -> val).average().orElse(0.0));
		res.add("muc (precision): " + this.mucP.stream().mapToDouble(val -> val).average().orElse(0.0));
		res.add("muc (f1): " + this.mucF1.stream().mapToDouble(val -> val).average().orElse(0.0));
		res.add("---");
		res.add("bcub (recall): " + this.bcubR.stream().mapToDouble(val -> val).average().orElse(0.0));
		res.add("bcub (precision): " + this.bcubP.stream().mapToDouble(val -> val).average().orElse(0.0));
		res.add("bcub (f1): " + this.bcubF1.stream().mapToDouble(val -> val).average().orElse(0.0));
		res.add("---");
		res.add("ceafm (recall): " + this.ceafmR.stream().mapToDouble(val -> val).average().orElse(0.0));
		res.add("ceafm (precision): " + this.ceafmP.stream().mapToDouble(val -> val).average().orElse(0.0));
		res.add("ceafm (f1): " + this.ceafmF1.stream().mapToDouble(val -> val).average().orElse(0.0));
		res.add("---");
		res.add("ceafe (recall): " + this.ceafeR.stream().mapToDouble(val -> val).average().orElse(0.0));
		res.add("ceafe (precision): " + this.ceafeP.stream().mapToDouble(val -> val).average().orElse(0.0));
		res.add("ceafe (f1): " + this.ceafeF1.stream().mapToDouble(val -> val).average().orElse(0.0));
		res.add("---");
		res.add("blanc (recall): " + this.blancR.stream().mapToDouble(val -> val).average().orElse(0.0));
		res.add("blanc (precision): " + this.blancP.stream().mapToDouble(val -> val).average().orElse(0.0));
		res.add("blanc (f1): " + this.blancF1.stream().mapToDouble(val -> val).average().orElse(0.0) + "\n");
		res.add("------------------------------> conll F1: " + this.conllF1);
		return res.toString();
	}
	public void incorporatePerformanceMetric(PerformanceMetric p) {
		tp += p.getTruePos();
		tn += p.getTrueNeg();
		fn += p.getFalseNeg();
		fp += p.getFalsePos();
	}
	public void setTruePos(double d) {
		tp = d;
	}
	public void setTrueNeg(double d) {
		tn = d;
	}
	public void setFalsePos(double d) {
		fp = d;
	}
	public void setFalseNeg(double d) {
		fn = d;
	}
	
	public void addTruePos(double d) {
		tp += d;
	}
	public void addTrueNeg(double d) {
		tn += d;
	}
	public void addFalsePos(double d) {
		fp += d;
	}
	public void addFalseNeg(double d) {
		fn += d;
	}
	
	public double getTruePos() {
		return tp;
	}
	public double getTrueNeg() {
		return tn;
	}
	public double getFalsePos() {
		return fp;
	}
	public double getFalseNeg() {
		return fn;
	}
	
	public double getRecall() {
		return tp*1.0/(tp+fn);
	}
	public double getPrecision() {
		return tp*1.0/(tp+fp);
	}
	public double getF1() {
		return 2*1.0*(getPrecision()*getRecall()/(getPrecision()+getRecall()));
	}
	public double getTrueNegRate() {
		return tp*1.0/(tp+fn);
	}
	public double getNegPredVal() {
		return tn*1.0 / (tn + fn);
	}
}
