package common;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.file.Paths;
import java.util.List;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import ecb_utils.ECBDoc;
import ecb_utils.ECBWrapper;
import edu.stanford.nlp.pipeline.ProtobufAnnotationSerializer;
import me.tongfei.progressbar.ProgressBar;

/**
 * Utility for serializing objects
 * @author acrem003
 *
 */
public class SerUtils {
	private static final Logger LOGGER = Logger.getLogger(SerUtils.class.getName());
	
	/**
	 * Generate a corenlp CoreDoc object for all ECB+ documents and serialize them
	 */
	public static void cacheSemanticParses() {
		Globals.USED_CACHED_PARSES = false;
		List<Integer> topics = IntStream.rangeClosed(1, 45).boxed().collect(Collectors.toList());
	    List<File> allFiles = ECBWrapper.getFilesFromTopics(topics);

		LOGGER.info("loading all files...");
		for(File f : ProgressBar.wrap(allFiles, "Load all files")) {
			File serExists = Paths.get(Globals.CACHED_CORE_DOCS.toString(), f.getName() + ".ser").toFile();
			if(serExists.exists())
				continue;
			ECBDoc doc = new ECBDoc(f);
			ProtobufAnnotationSerializer ser = new ProtobufAnnotationSerializer();
			serializeObj(ser.toProto(doc.coreDoc.annotation()), Paths.get(Globals.CACHED_CORE_DOCS.toString(), f.getName() + ".ser").toFile());
		}

		LOGGER.info("cache built ");
		System.exit(0);
		
	}
	
	/**
	 * Serialize an object to a location
	 * @param o: object to serialize
	 * @param loc: location
	 */
	public static void serializeObj(Object o, File loc) {
		try {
			FileOutputStream fos = new FileOutputStream(loc);
			ObjectOutputStream oos = new ObjectOutputStream(fos);
			oos.writeObject(o);
			oos.close();
			fos.close();
		} catch (IOException ioe) {
			ioe.printStackTrace();
		}
	}
	
	/**
	 * Load a previously serialized object
	 * @param path: where the serialized object is stored
	 * @return
	 */
	public static Object loadObj(File path) {
		
		Object obj = null;
		try {
			// Reading the object from a file
			FileInputStream file = new FileInputStream(path);
			ObjectInputStream in = new ObjectInputStream(file);

			// Method for deserialization of object
			obj = in.readObject();

			in.close();
			file.close();
		}

		catch (IOException | ClassNotFoundException ex) {
			System.out.println(ex);
		}
		
		return obj;
  
    } 

}
