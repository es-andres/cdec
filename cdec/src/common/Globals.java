package common;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

//import org.apache.spark.sql.SparkSession;

import com.google.common.graph.GraphBuilder;
import com.google.common.graph.Graphs;
import com.google.common.graph.MutableGraph;

import edu.emory.clir.clearnlp.dependency.DEPNode;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.CoreDocument;
import edu.stanford.nlp.pipeline.CoreNLPProtos;
import naf.EventNode;
import naf.NafDoc;
import vu.wntools.wordnet.WordnetLmfSaxParser;

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
	public static HashMap<String, NafDoc> nafDocs;
	public static HashMap<String, ArrayList<String>> globalActionToMentions = new HashMap<String, ArrayList<String>>();
	public static final HashMap<String, HashSet<String>> cleanSentences = DataWriter.cleanSentenceDict(CLEAN_SENT_PATH.toFile());
	public static HashSet<HashSet<EventNode>> truePairsInTrain = new HashSet<HashSet<EventNode>>();
	public static WordnetLmfSaxParser wn_parser = new WordnetLmfSaxParser(Globals.WN_LMF_PATH);
	public static HashMap<String, CoreNLPProtos.Document> cachedCoreDocs = new HashMap<String, CoreNLPProtos.Document>();
	public static HashMap<String, HashMap<String, DEPNode[]>> cachedSRL = new HashMap<String, HashMap<String, DEPNode[]>>();
	public static final HashSet<String> strIgnore = new HashSet<String>(Arrays.asList("nt", "t"));


	
	
	//3
	// taken from Vossen and Cybulska pg. 8
	// "24","25","26","27","28","29","30","31","32","33","34","35",
	public static final HashSet<String> TEST_TOPICS = new HashSet<String>(Arrays.asList("36", "37", "38", "39", "40", "41", "42", "43", "44", "45"));
	public static final HashSet<String> DEV_TOPICS = new HashSet<String>(Arrays.asList("44","45"));
	// split arff into train/test
	public static final boolean SPLIT_ARFF = true; 
	
	// WD
	public static final String WD_TRAIN_FILE = RESULTS_DIR + "/WD_cd_vectors_train.arff";
	public static final String WD_TEST_FILE = RESULTS_DIR + "/WD_cd_vectors_test.arff";
	public static final String WD_TEST_INDEX_FILE = RESULTS_DIR + "/WD_cd_vectors_test_index.arff";
	public static final String WD_CLASSIFIER_PATH = "/home/acrem003/Documents/Cognac/svn/projects/onr/code/java/new_classifiers";
//	trigger_sim_vec|trigger_wn_sim|trigger_sim_strict|trigger_sts|event_sts
	public static final String WD_FILTERED_ATTRIBS = "trigger_sim_vec|trigger_wn_sim|trigger_sim_strict|trigger_sts|event_sts|event_sim_strict|context_sts|coreferent";
	// CD
//	CD_vectors_topic_train.arff
	public static final String CD_TRAIN_FILE = RESULTS_DIR + "/CD_vectors_train.arff";
	public static final String CD_TEST_FILE = RESULTS_DIR + "/CD_vectors_test.arff";
	public static final String CD_TEST_INDEX_FILE = RESULTS_DIR + "/CD_vectors_index_test.arff";
	public static final String CD_CLASSIFIER_PATH = "/home/acrem003/Documents/Cognac/svn/projects/onr/code/java/new_classifiers";
	
	public static final String WN_LMF_PATH = "/home/acrem003/Documents/Cognac/outside_code/vua-resources/wneng-30.lmf.xml.xpos.gz";
	public static final String CONLL_OUTPUT_DIR = "/home/acrem003/Documents/Cognac/svn/projects/onr/code/java/conll_results";
	
	public static final boolean TOKENIZE = false;
	public static final boolean VERBOSE = true;
	public static final boolean AUGMENTED = true;
	public static final boolean DOING_DEV = false;
	private static String[] stopwords = {"a", "as", "able", "about", "above", "according", "accordingly", 
										 "across", "actually", "after", "afterwards", "again", "against", 
										 "aint", "all", "allow", "allows", "almost", "alone", "along", "already", "also", 
										 "although", "always", "am", "among", "amongst", "an", "and", "another", "any", "anybody", "anyhow", "anyone", "anything", "anyway", "anyways", "anywhere", "apart", "appear", "appreciate", "appropriate", "are", "arent", "around", "as", "aside", "ask", "asking", "associated", "at", "available", "away", "awfully", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being", "believe", "below", "beside", "besides", "best", "better", "between", "beyond", "both", "brief", "but", "by", "cmon", "cs", "came", "can", "cant", "cannot", "cant", "cause", "causes", "certain", "certainly", "changes", "clearly", "co", "com", "come", "comes", "concerning", "consequently", "consider", "considering", "contain", "containing", "contains", "corresponding", "could", "couldnt", "course", "currently", "definitely", "described", "despite", "did", "didnt", "different", "do", "does", "doesnt", "doing", "dont", "done", "down", "downwards", "during", "each", "edu", "eg", "eight", "either", "else", "elsewhere", "enough", "entirely", "especially", "et", "etc", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "far", "few", "ff", "fifth", "first", "five", "followed", "following", "follows", "for", "former", "formerly", "forth", "four", "from", "further", "furthermore", "get", "gets", "getting", "given", "gives", "go", "goes", "going", "gone", "got", "gotten", "greetings", "had", "hadnt", "happens", "hardly", "has", "hasnt", "have", "havent", "having", "he", "hes", "hello", "help", "hence", "her", "here", "heres", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "hi", "him", "himself", "his", "hither", "hopefully", "how", "howbeit", "however", "i", "id", "ill", "im", "ive", "ie", "if", "ignored", "immediate", "in", "inasmuch", "inc", "indeed", "indicate", "indicated", "indicates", "inner", "insofar", "instead", "into", "inward", "is", "isnt", "it", "itd", "itll", "its", "its", "itself", "just", "keep", "keeps", "kept", "know", "knows", "known", "last", "lately", "later", "latter", "latterly", "least", "less", "lest", "let", "lets", "like", "liked", "likely", "little", "look", "looking", "looks", "ltd", "mainly", "many", "may", "maybe", "me", "mean", "meanwhile", "merely", "might", "more", "moreover", "most", "mostly", "much", "must", "my", "myself", "name", "namely", "nd", "near", "nearly", "necessary", "need", "needs", "neither", "never", "nevertheless", "new", "next", "nine", "no", "nobody", "non", "none", "noone", "nor", "normally", "not", "nothing", "novel", "now", "nowhere", "obviously", "of", "off", "often", "oh", "ok", "okay", "old", "on", "once", "one", "ones", "only", "onto", "or", "other", "others", "otherwise", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "own", "particular", "particularly", "per", "perhaps", "placed", "please", "plus", "possible", "presumably", "probably", "provides", "que", "quite", "qv", "rather", "rd", "re", "really", "reasonably", "regarding", "regardless", "regards", "relatively", "respectively", "right", "said", "same", "saw", "say", "saying", "says", "second", "secondly", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "shall", "she", "should", "shouldnt", "since", "six", "so", "some", "somebody", "somehow", "someone", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "specified", "specify", "specifying", "still", "sub", "such", "sup", "sure", "ts", "take", "taken", "tell", "tends", "th", "than", "thank", "thanks", "thanx", "that", "thats", "thats", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "theres", "thereafter", "thereby", "therefore", "therein", "theres", "thereupon", "these", "they", "theyd", "theyll", "theyre", "theyve", "think", "third", "this", "thorough", "thoroughly", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "took", "toward", "towards", "tried", "tries", "truly", "try", "trying", "twice", "two", "un", "under", "unfortunately", "unless", "unlikely", "until", "unto", "up", "upon", "us", "use", "used", "useful", "uses", "using", "usually", "value", "various", "very", "via", "viz", "vs", "want", "wants", "was", "wasnt", "way", "we", "wed", "well", "were", "weve", "welcome", "well", "went", "were", "werent", "what", "whats", "whatever", "when", "whence", "whenever", "where", "wheres", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whos", "whoever", "whole", "whom", "whose", "why", "will", "willing", "wish", "with", "within", "without", "wont", "wonder", "would", "would", "wouldnt", "yes", "yet", "you", "youd", "youll", "youre", "youve", "your", "yours", "yourself", "yourselves", "zero"};
	public static final HashSet<String> STOP_WORDS = new HashSet<String>(Arrays.asList(stopwords));
}