package event_clustering;
import java.util.Arrays;
import java.util.List;

import common.Globals;
import de.lmu.ifi.dbs.elki.data.NumberVector;
import de.lmu.ifi.dbs.elki.data.type.SimpleTypeInformation;
import de.lmu.ifi.dbs.elki.data.type.VectorFieldTypeInformation;
import de.lmu.ifi.dbs.elki.distance.similarityfunction.AbstractVectorSimilarityFunction;
import ecb_utils.ECBDoc;
import ecb_utils.ECBWrapper;
import ecb_utils.EventNode;
import feature_extraction.EvPairDataset;
import weka.classifiers.Classifier;
import weka.core.Instance;

public class ElkiDistanceFunction extends AbstractVectorSimilarityFunction{
	
	Classifier clf;
	EvPairDataset vectorizer;
	ECBWrapper dataWrapper;
	
	public ElkiDistanceFunction(Classifier clf, EvPairDataset vectorizer, ECBWrapper dataWrapper) {
		this.clf = clf;
		this.vectorizer = vectorizer;
		this.dataWrapper = dataWrapper;
	}
	
	@Override
	public double similarity(NumberVector arr1, NumberVector arr2) {
        EventNode temp = ElkiCluster.arrToEv(arr1.toArray(), "UNK");
        ECBDoc doc = dataWrapper.docs.get(temp.file.getName());
        String ev_id = doc.toks.get(doc.mentionIdToTokens.get(temp.m_id).get(0)).get("ev_id");
		EventNode ev1 = ElkiCluster.arrToEv(arr1.toArray(), ev_id);
        temp = ElkiCluster.arrToEv(arr2.toArray(), "UNK");
        doc = dataWrapper.docs.get(temp.file.getName());
        ev_id = doc.toks.get(doc.mentionIdToTokens.get(temp.m_id).get(0)).get("ev_id");
		EventNode ev2 = ElkiCluster.arrToEv(arr2.toArray(), ev_id);

		List<EventNode> pair = Arrays.asList(ev1, ev2);
		Instance inst = this.vectorizer.evPairVector(dataWrapper, pair, Globals.LEMMATIZE, Globals.POS);
		double dist = 0;
		try {
			dist = this.clf.distributionForInstance(inst)[1];
		} catch (Exception e) {
			e.printStackTrace();
		}

		return 1 - dist;
	}
	
	  @Override
	  public SimpleTypeInformation<? super NumberVector> getInputTypeRestriction() {
	    return VectorFieldTypeInformation.typeRequest(NumberVector.class, 4, 4);
	    // alternative:
	    // return TypeUtil.NUMBER_VECTOR_FIELD_2D;
	  }

}
