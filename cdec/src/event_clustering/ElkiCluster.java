package event_clustering;

import static org.junit.Assert.assertEquals;

import java.io.File;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

import common.Globals;
import de.lmu.ifi.dbs.elki.algorithm.clustering.affinitypropagation.AffinityPropagationClusteringAlgorithm;
import de.lmu.ifi.dbs.elki.algorithm.clustering.affinitypropagation.SimilarityBasedInitializationWithMedian;
import de.lmu.ifi.dbs.elki.data.Cluster;
import de.lmu.ifi.dbs.elki.data.Clustering;
import de.lmu.ifi.dbs.elki.data.NumberVector;
import de.lmu.ifi.dbs.elki.data.model.MedoidModel;
import de.lmu.ifi.dbs.elki.data.type.TypeUtil;
import de.lmu.ifi.dbs.elki.database.Database;
import de.lmu.ifi.dbs.elki.database.StaticArrayDatabase;
import de.lmu.ifi.dbs.elki.database.ids.DBIDIter;
import de.lmu.ifi.dbs.elki.database.ids.DBIDRange;
import de.lmu.ifi.dbs.elki.database.relation.Relation;
import de.lmu.ifi.dbs.elki.datasource.ArrayAdapterDatabaseConnection;
import de.lmu.ifi.dbs.elki.datasource.DatabaseConnection;
import de.lmu.ifi.dbs.elki.logging.LoggingConfiguration;

import ecb_utils.ECBDoc;
import ecb_utils.ECBWrapper;
import ecb_utils.EventNode;
import feature_extraction.EvPairDataset;
import weka.classifiers.Classifier;

public class ElkiCluster {
	
	public static EventNode arrToEv(double[] arr, String ev_id) {
		String fName = "";
		// topic
		fName += ((int)arr[0]) + "_" + ((int)arr[1]);
		fName += (arr[2] == 1.0) ? "ecbplus_aug.en.naf" : "ecb_aug.en.naf";
		File file = Paths.get(Globals.ECBPLUS_DIR.toString(), fName).toFile();
		EventNode ev = new EventNode(file, String.valueOf(((int)arr[3])), ev_id);
		
		return ev;
	}
	public double[][] makeData(List<EventNode> evs){
		/*
		 * 0: topic
		 * 1: file_num
		 * 2: 0 -> ecb; 1 -> ecbplus
		 * 3: m_id 
		 * 
		 */
		HashMap<Integer, Double> keys = new HashMap<Integer, Double>();

		double[][] data = new double[evs.size()][4];
		for(int i = 0; i < evs.size(); i++) {
			String[] file = evs.get(i).file.getName().split("_");
			double topic = Double.parseDouble(file[0]);
			double file_num = Double.parseDouble(file[1].substring(0, file[1].indexOf("e")));
			String sub = file[1].substring(file[1].indexOf("e"), file[1].length());
			double sub_topic = sub.contains("plus") ? 1.0 : 0.0;
			double m_id = Double.parseDouble(evs.get(i).m_id);
			keys.put(0, topic);
			keys.put(1, file_num);
			keys.put(2, sub_topic);
			keys.put(3, m_id);
			for(int j = 0; j < data[i].length; j++) {
				data[i][j] = keys.get(j);
			}
			assertEquals(evs.get(i), arrToEv(data[i], evs.get(i).ev_id));
		}
		
		return data;
	} 
	public ArrayList<HashSet<EventNode>> init(List<EventNode> evs, Classifier clf, EvPairDataset vectorizer, ECBWrapper dataWrapper) {
		// put this in main to get it to work
//		Elki elki = new Elki();
//		List<EventNode> evs = dataWrapper.getEventSet(testPairs, testDocClusters.get(c_id).split(" "));
//		cdecChains.addAll(elki.init(evs, clf, dataMaker, dataWrapper));
//		clusterDocs.addAll(Arrays.asList(testDocClusters.get(c_id).split(" ")));
		LoggingConfiguration.setStatistics();
		

	    // Generate a random data set.
	    // Note: ELKI has a nice data generator class, use that instead.
	    double[][] data = makeData(evs);
		
		DatabaseConnection dbc = new ArrayAdapterDatabaseConnection(data);
	    // Create a database (which may contain multiple relations!)
	    Database db = new StaticArrayDatabase(dbc, null);
	    
	    // Load the data into the database (do NOT forget to initialize...)
	    db.initialize();

	    // Relation containing the number vectors:
	    Relation<NumberVector> rel = db.getRelation(TypeUtil.NUMBER_VECTOR_FIELD);
	    // We know that the ids must be a continuous range:
	    DBIDRange ids = (DBIDRange) rel.getDBIDs();

	    ElkiDistanceFunction dist = new ElkiDistanceFunction(clf, vectorizer, dataWrapper);
	    SimilarityBasedInitializationWithMedian<NumberVector> sim = new SimilarityBasedInitializationWithMedian<NumberVector>(dist,  0.95);
	    
	    AffinityPropagationClusteringAlgorithm<NumberVector> clust = new AffinityPropagationClusteringAlgorithm<NumberVector>(sim, 0.05, 15, 200);
	    Clustering<MedoidModel> c = clust.run(db);
	    
	    
	    // Output all clusters:
	    ArrayList<HashSet<EventNode>> chains = new ArrayList<HashSet<EventNode>>();
	    for(Cluster<MedoidModel> clu : c.getAllClusters()) {

	      HashSet<EventNode> chain = new HashSet<EventNode>();
	      for(DBIDIter it = clu.getIDs().iter(); it.valid(); it.advance()) {
	        // To get the vector use:
	        // NumberVector v = rel.get(it);

	        // Offset within our DBID range: "line number"
	        final int offset = ids.getOffset(it);
	        EventNode temp = arrToEv(data[offset], "UNK");
	        ECBDoc doc = dataWrapper.docs.get(temp.file.getName());
	        String ev_id = doc.toks.get(doc.mentionIdToTokens.get(temp.m_id).get(0)).get("ev_id");
	        chain.add(arrToEv(data[offset], ev_id));

	        // Do NOT rely on using "internalGetIndex()" directly!
	      }
	      chains.add(chain);
	    }
	    return chains;
	}
	
}
