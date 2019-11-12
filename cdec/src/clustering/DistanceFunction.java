package clustering;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

import common.ECBWrapper;
import common.Globals;
import comparer.EvPairDataset;
import de.lmu.ifi.dbs.elki.data.NumberVector;
import de.lmu.ifi.dbs.elki.data.type.SimpleTypeInformation;
import de.lmu.ifi.dbs.elki.data.type.VectorFieldTypeInformation;
import de.lmu.ifi.dbs.elki.distance.distancefunction.AbstractNumberVectorDistanceFunction;
import de.lmu.ifi.dbs.elki.distance.similarityfunction.AbstractVectorSimilarityFunction;
import naf.EventNode;
import naf.NafDoc;
import weka.classifiers.Classifier;
import weka.core.Instance;

public class DistanceFunction extends AbstractVectorSimilarityFunction{
	
	Classifier clf;
	EvPairDataset vectorizer;
	ECBWrapper dataWrapper;
	
	public DistanceFunction(Classifier clf, EvPairDataset vectorizer, ECBWrapper dataWrapper) {
		this.clf = clf;
		this.vectorizer = vectorizer;
		this.dataWrapper = dataWrapper;
	}
	
	@Override
	public double similarity(NumberVector arr1, NumberVector arr2) {
        EventNode temp = Elki.arrToEv(arr1.toArray(), "UNK");
        NafDoc doc = dataWrapper.docs.get(temp.file.getName());
        String ev_id = doc.toks.get(doc.mentionIdToTokens.get(temp.m_id).get(0)).get("ev_id");
		EventNode ev1 = Elki.arrToEv(arr1.toArray(), ev_id);
        temp = Elki.arrToEv(arr2.toArray(), "UNK");
        doc = dataWrapper.docs.get(temp.file.getName());
        ev_id = doc.toks.get(doc.mentionIdToTokens.get(temp.m_id).get(0)).get("ev_id");
		EventNode ev2 = Elki.arrToEv(arr2.toArray(), ev_id);

		List<EventNode> pair = Arrays.asList(ev1, ev2);
		Instance inst = this.vectorizer.evPairVector(dataWrapper, pair, Globals.LEMMATIZE, Globals.POS);
		double dist = 0;
		try {
			dist = this.clf.distributionForInstance(inst)[1];
		} catch (Exception e) {
			e.printStackTrace();
		}
		// checking this to make sure elki is working
		// ... so it mostly works but there's the "noise" cluster that has mistakes?
//		double dist = ev1.corefers(ev2) ? 1.0 : 0.0;
		return 1 - dist;
	}
	
	  @Override
	  public SimpleTypeInformation<? super NumberVector> getInputTypeRestriction() {
	    return VectorFieldTypeInformation.typeRequest(NumberVector.class, 4, 4);
	    // alternative:
	    // return TypeUtil.NUMBER_VECTOR_FIELD_2D;
	  }

}
