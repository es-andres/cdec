package common;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

import com.google.common.graph.MutableGraph;

import ecb_utils.ECBDoc;
import ecb_utils.EventNode;
import edu.emory.clir.clearnlp.dependency.DEPNode;
import edu.stanford.nlp.pipeline.CoreNLPProtos;

public class Globals {

	/*
	 * shared constants
	 */
	public static final Path ROOT_DIR = Paths.get(System.getProperty("user.dir")).getParent();
	public static final Path ECBPLUS_DIR = Paths.get(ROOT_DIR.toString(), "data", "ecb_plus");
	public static final Path RESULTS_DIR = Paths.get(ROOT_DIR.toString(), "data", "results");
	public static final Path PYTHON_DIR = Paths.get(ROOT_DIR.toString(), "python_assets");
	public static final Path CONLL_DIR = Paths.get(ROOT_DIR.toString(), "data", "conll_files");
	public static final Path CONLL_SCORER_PATH = Paths.get(ROOT_DIR.toString(), "perl_assets", "reference-coreference-scorers", "scorer.pl");
	public static final Path PY_DOC_CLUSTER = Paths.get(PYTHON_DIR.toString(), "clustering", "cluster_documents.sh");
	public static final Path CLEAN_SENT_PATH = Paths.get(ROOT_DIR.toString(), "data", "ECBplus_coreference_sentences.csv");
	public static final Path CACHED_CORE_DOCS = Paths.get(ROOT_DIR.toString(), "data", "cached_core_docs");
	public static final Path CACHED_SRL = Paths.get(ROOT_DIR.toString(), "data", "cached_srl");
	public static final String DELIM = "DELIM";
	public static final List<Integer> DEAD_TOPICS = Arrays.asList(15, 17);
	public static final Path CORE_NLP_SERVER = Paths.get("/home/acrem003/Documents/Cognac/stanford-corenlp-full-2018-01-31");
	public static final String W2V_SERVER = "http://localhost:8000";
	public static final boolean LEMMATIZE = true;
	public static final String[] POS = {"N", "V", "R", "J", "CD"};
	
	
	public static HashSet<Integer> trainTopics;
	public static HashSet<Integer> testTopics;
	public static boolean USED_CACHED_PARSES = true;
	
	/*
	 * shared data structures
	 */
	public static HashMap<String, MutableGraph<EventNode>> globalCorefGraphs = new HashMap<String, MutableGraph<EventNode>>();
	public static HashMap<String, HashSet<HashSet<EventNode>>> globalCorefChains = new HashMap<String, HashSet<HashSet<EventNode>>>();
	public static HashMap<String, ECBDoc> nafDocs;
	public static HashMap<String, ArrayList<String>> globalActionToMentions = new HashMap<String, ArrayList<String>>();
	public static final HashMap<String, HashSet<String>> cleanSentences = DataWriter.cleanSentenceDict(CLEAN_SENT_PATH.toFile());
	public static HashSet<HashSet<EventNode>> truePairsInTrain = new HashSet<HashSet<EventNode>>();
	public static HashMap<String, CoreNLPProtos.Document> cachedCoreDocs = new HashMap<String, CoreNLPProtos.Document>();
	public static HashMap<String, HashMap<String, DEPNode[]>> cachedSRL = new HashMap<String, HashMap<String, DEPNode[]>>();
	public static final HashSet<String> strIgnore = new HashSet<String>(Arrays.asList("nt", "t"));

	

}