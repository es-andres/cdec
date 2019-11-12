package weka;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.LinkedList;
import java.util.List;
import java.util.Vector;
import java.util.regex.Pattern;

import common.GeneralTuple;
import common.Globals;
import comparer.SimilarityVector;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.CVParameterSelection;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.SerializationHelper;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.MultiFilter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.RemoveByName;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.SubsetByExpression;

/**
 * Adopted from Weka demo by FracPete (fracpete at waikato dot ac dot nz)
 * 
 * @author AndresCremisini
 */
public class Weka {

	public RandomForest classifier = null;
	protected MultiFilter filter = null;
	protected Instances train_set = null;
	protected Instances test_set = null;
	protected Evaluation evaluation = null;
	protected CVParameterSelection optim = null;
	String classifierType;
	private Normalize norm;
	private boolean subsetAttributes;
	

	public Weka(boolean subsetAttributes, String classifierType, String trainPath, String testPath) {
		super();
		this.subsetAttributes = subsetAttributes;
		
		try {
			// set filter
			filter = setFilter();
			
			// set classifier
			this.classifierType = classifierType;
			String[] options = {"-attribute-importance"};
			classifier = new RandomForest();
			classifier.setOptions(options);

			// set optimization options 
			optim = new CVParameterSelection();
			optim.setClassifier(classifier);
			optim.setNumFolds(2);
			// bag size (%) 
			optim.addCVParameter("P 100 100 1 R");
			// number of trees 
			optim.addCVParameter("I 500 500 1 R");
			// number of attributes to randomly investigate
//			optim.addCVParameter("K 0 10 3 R");
			// minimum number of instances per leaf
			optim.addCVParameter("M 10 10 1 R");
			// depth of tree
			optim.addCVParameter("depth 64 64 1 R");
			Object[] params = optim.getCVParameters();
			System.out.println("Will tune, using following parameters and ranges:");
			for(Object o : params)
				System.out.println(o);
			
			// init train
			train_set = initAndFilterData(trainPath);
			// init test
			test_set = initAndFilterData(testPath);

		} 
		catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public void trainClassifier() {

		try {
			// train
			System.out.println("Training classifier...");
//			optim.buildClassifier(train_set);
			String[] options = {"-P", "100", "-I", "500", "-M", "10", "-depth", "64", "-attribute-importance"};
//			classifier.setOptions(optim.getBestClassifierOptions());
			classifier.setOptions(options);
			for(String o : classifier.getOptions())
				System.out.println(o);
			classifier.buildClassifier(train_set);
			// evaluate
			System.out.println("\t\t\t\t\t===================\n\t\t\t\t\t   Training eval\n\t\t\t\t\t===================\n");
			evaluateClassifier(train_set);
			
		}
		catch(Exception e) {
			e.printStackTrace();
		}
	}
	
	public void testClassifier() {
		System.out.println("\t\t\t\t\t===================\n\t\t\t\t\t   Testing eval\n\t\t\t\t\t===================\n");
		evaluateClassifier(test_set);
	}
	
	public void saveClassifierAndFilter(String saveDir, String classifierName) {
		
		// save
		File modelSavePath = new File(saveDir + "/" + classifierName  + ".model");
		File filterSavePath = new File(saveDir + "/" + classifierName  + ".filter");
		try {
			weka.core.SerializationHelper.write(modelSavePath.getPath(), classifier);
			weka.core.SerializationHelper.write(filterSavePath.getPath(), filter);
		} 
		catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
	public void loadClassifierAndFilter(String classifierPath) {
		try {
			// load classifier
			this.classifier = (RandomForest) SerializationHelper.read(new FileInputStream(classifierPath));
			this.filter = (MultiFilter) SerializationHelper.read(new FileInputStream(classifierPath.replace(".model", ".filter")));
			this.norm = (Normalize)this.filter.getFilters()[this.filter.getFilters().length - 1];
		}
		catch(Exception e) {
			e.printStackTrace();
		}
	}
	
	private Instances initAndFilterData(String path) {
		Instances data = null;
		try {
			data = new Instances(new BufferedReader(new FileReader(path)));
			data.setClassIndex(data.numAttributes() - 1);
			filter.setInputFormat(data);
			data = Filter.useFilter(data, filter);
		}
		catch(Exception e) {
			e.printStackTrace();
		}
		return data;
	}
	

	private MultiFilter setFilter() throws Exception {

		MultiFilter filter = new MultiFilter();
		RemoveByName attribFilter = null;
		if(subsetAttributes) {
			attribFilter = new RemoveByName();
			attribFilter.setInvertSelection(true);
			attribFilter.setExpression(Pattern.compile(Globals.WD_FILTERED_ATTRIBS).toString());
		}

		Randomize randomFilter = new Randomize();

		Normalize normalizeFilter = new Normalize();

		if(subsetAttributes) {
			((MultiFilter)filter).setFilters(new Filter[] { attribFilter, randomFilter, normalizeFilter });
		}
		else {
			((MultiFilter)filter).setFilters(new Filter[] {randomFilter, normalizeFilter });
		}
		
		return filter;

	}
	
	private void evaluateClassifier(Instances data) {
		
		try {
			evaluation = new Evaluation(data);
			evaluation.evaluateModel(classifier, data);
			System.out.println(this.toString());
		} 
		catch (Exception e) {
			e.printStackTrace();
		}
	}



	public GeneralTuple<String,Double> classifyInstance(String vec){

		
		Enumeration<Attribute> attributes = train_set.enumerateAttributes();
		String[] vector = vec.split(",");
		Instance instance = new DenseInstance(train_set.numAttributes());

		int i = 0;

		while(attributes.hasMoreElements()) {
			if(vector[i].contains("true") || vector[i].contains("false"))
				instance.setValue(attributes.nextElement(), vector[i]);
			else
				instance.setValue(attributes.nextElement(), Double.parseDouble(vector[i]));
			i++;
		}
		instance.setDataset(train_set);
		try {
			norm.input(instance);
			
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		instance = norm.output();
		
		double[] dist = null;
		double pred_idx = 0;
		try {
			pred_idx = classifier.classifyInstance(instance);
			dist = classifier.distributionForInstance(instance);
		}
		catch(Exception e) {
			e.printStackTrace();
		}
		String predString = train_set.classAttribute().value((int)pred_idx);
		System.out.println(pred_idx);
		System.out.println(predString);
		System.out.println(dist[0] + " " + dist[1]);

		double y_hat = predString.equals("true") ? dist[0] : dist[1];

		//TODO: this breaks index file for CDEC (so maybe just need to generate everything?)
		// hardwire threshold
		// ~around 0.4 seems to be best
		// thresh, conf, conll
		// 0.4, 0.6 -> 83.116666 (wdec)
		// 0.67 -> 68.266 (wdec cd)
		if(y_hat > 0.6) {
			predString = "true";
		}
		else {
			predString = "false";
		}


		return new GeneralTuple<String,Double>(predString, y_hat);
	}

	/**
	 * outputs some data about the classifier
	 */
	@Override
	public String toString() {
		StringBuffer result;

		result = new StringBuffer();
		result.append("Weka - \n===========\n\n");

		result.append("Classifier...: " + Utils.toCommandLine(classifier) + "\n");
		if (filter instanceof OptionHandler) {
			result.append("Filter.......: " + filter.getClass().getName() + " "
					+ Utils.joinOptions(((OptionHandler) filter).getOptions()) + "\n");
		} else {
			result.append("Filter.......: " + filter.getClass().getName() + "\n");
		}
//		result.append("Training file: " + training_file + "\n");
//		result.append("\n");

		result.append(classifier.toString() + "\n");
		result.append(evaluation.toSummaryString() + "\n");
		try {
			result.append(evaluation.toMatrixString() + "\n");
		} catch (Exception e) {
			e.printStackTrace();
		}
		try {
			result.append(evaluation.toClassDetailsString() + "\n");
		} catch (Exception e) {
			e.printStackTrace();
		}

		return result.toString();
	}

}
