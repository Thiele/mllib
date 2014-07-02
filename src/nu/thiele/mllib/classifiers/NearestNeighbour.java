package nu.thiele.mllib.classifiers;

import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import nu.thiele.mllib.data.Data.DataEntry;
import nu.thiele.mllib.utils.Utils;

/**
 * 
 * @author Andreas Thiele
 *
 *
 * An implementation of knn.
 * Uses Euclidean distance weighted by 1/distance
 * 
 * Main method to classify if entry is male or female based on:
 * Height, weight
 */
public class NearestNeighbour implements IClassifier{
	private int k;
	private List<DataEntry> dataSet;
	private DistanceWeighing weight = DistanceWeighing.INDICATOR;
	
	
	/**
	 * 
	 * @param dataSet The set. I assume that everything is fine with regards to feature vector
	 * @param k The number of neighbours to use
	 */
	public NearestNeighbour(List<DataEntry> dataSet, int k){
		this.k = k;
		this.dataSet = dataSet;
	}
	
	public static DataEntry[] getKNearestNeighbours(List<DataEntry> dataSet, double[] x, int k){
		DataEntry[] retur = new DataEntry[k];
		double fjernest = Double.MIN_VALUE;
		int index = 0;
		int i = 0;
		for(DataEntry tse : dataSet){
			double distance = distance(x,tse.getX());
			if(i < retur.length){ //Hvis ikke fyldt
				if(retur[i] == null){
					retur[i] = tse;
					if(distance > fjernest){
						index = i;
						fjernest = distance;
					}
				}
			}
			else{
				if(distance < fjernest){
					retur[index] = tse;
					double f = 0.0f;
					int ind = 0;
					for(int j = 0; j < retur.length; j++){
						double dt = distance(retur[j].getX(),x);
						if(dt > f){
							f = dt;
							ind = j;
						}
					}
					fjernest = f;
					index = ind;
				}
			}
			i++;
		}
		return retur;
	}
	
	private DataEntry[] getNearestNeighbourType(double[] x){
		return NearestNeighbour.getKNearestNeighbours(this.dataSet, x, this.k);
	}
	
	/**
	 * Computes Euclidean distance
	 * @param a From
	 * @param b To
	 * @return Distance 
	 */
	public static double distance(double[] a, double[] b){
		double distance = 0.0f;
		for(int i = 0; i < a.length; i++){
			double t = a[i]-b[i];
			distance = distance+t*t;
		}
		return (double) Math.sqrt(distance);
	}
	/**
	 * 
	 * @param e Entry to be classifies
	 * @return The class of the most probable class
	 */
	public Object classify(double[] x){
		Map<Object, Double> classcount = this.calculateProbabilityForClassifications(x);
		//Find right choice
		Object o = null;
		double max = 0;
		HashMap<String,Object> toStringMap = new HashMap<String,Object>();
		for(Object ob : classcount.keySet()){
			toStringMap.put(ob.toString(), ob);
			if(classcount.get(ob) > max){
				max = classcount.get(ob);
				o = ob;
			}
		}
		LinkedList<String> stringRepList = new LinkedList<String>(toStringMap.keySet());
		Collections.sort(stringRepList);
		if(stringRepList.size() == 0) return o;
		//Return first to make it deterministic. Should perhaps be random choice instead
		return toStringMap.get(stringRepList.get(0));
	}
	
	private double getWeight(double input){
		switch(this.weight){
			default:
			case INDICATOR: return 1;
			case INVERSE: return 1.0f/input; 
		}
	}

	@Override
	public void loadClassifier() {} //Lazy classifier, do nothing...

	@Override
	public void setTrainingData(List<DataEntry> data) {
		this.dataSet = data;
	}
	
	public void setWeighing(DistanceWeighing w){
		this.weight = w;
	}
	
	public static enum DistanceWeighing{
		INDICATOR, INVERSE
	}

	@Override
	public Map<Object, Double> calculateProbabilityForClassifications(double[] x) {
		HashMap<Object,Double> classcount = new HashMap<Object,Double>();
		DataEntry[] de = this.getNearestNeighbourType(x);
		for(int i = 0; i < de.length; i++){
			double weight = this.getWeight(distance(x, de[i].getX())); 
			if(!classcount.containsKey(de[i].getY())){
				classcount.put(de[i].getY(), weight);
			}
			else{
				classcount.put(de[i].getY(), classcount.get(de[i].getY())+weight);
			}
		}
		Utils.percentify(classcount);
			
		return classcount;
	}
}