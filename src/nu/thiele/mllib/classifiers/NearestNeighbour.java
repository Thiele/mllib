package nu.thiele.mllib.classifiers;

import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

import nu.thiele.mllib.data.Data.DataEntry;
import nu.thiele.mllib.utils.Utils;

/**
 * 
 * @author Andreas Thiele
 *
 *
 * An implementation of knn.
 * 
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
	public NearestNeighbour(int k){
		this.k = k;
	}
	
	public NearestNeighbour(){
		this(10);
	}
	
	public static DataEntry[] getKNearestNeighbours(List<DataEntry> dataSet, final double[] x, int k){
		Comparator<DataEntry> distanceComparator = new Comparator<DataEntry>(){
			@Override
			public int compare(DataEntry arg0, DataEntry arg1) {
				double distA = distance(arg0.getX(), x);
				double distB = distance(arg1.getX(), x);
				if(distA < distB) return 1;
				else if(distA == distB) return 0;
				return -1;
			}
		};
		PriorityQueue<DataEntry> pe = new PriorityQueue<DataEntry>(k, distanceComparator);
		double fjernest = Double.MIN_VALUE;
		int i = 0;
		for(DataEntry tse : dataSet){
			if(i < k){ //Not full
				pe.add(tse);
				fjernest = distance(pe.peek().getX(), x);
			}
			else{
				double distance = distance(x,tse.getX());
				if(distance < fjernest){
					pe.poll();
					pe.add(tse);
					fjernest = distance(pe.peek().getX(),x);
				}
			}
			i++;
		}
		DataEntry[] retur = new DataEntry[k];
		i = 0;
		for(DataEntry d : pe){
			retur[i] = d;
			i++;
		}
		return retur;
	}
	
	private DataEntry[] getNearestNeighbourType(double[] x){
		int n = this.k;
		if(this.dataSet.size() < this.k) n = this.dataSet.size();
		return NearestNeighbour.getKNearestNeighbours(this.dataSet, x, n);
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
	public double classify(double[] x){
		Map<Double, Double> classcount = this.probability(x);
		//Find right choice
		double o = -1;
		double max = 0;
		HashMap<String,Double> toStringMap = new HashMap<String,Double>();
		for(Double ob : classcount.keySet()){
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
	public void train(double[][] x, double[] y) {
		this.dataSet = new LinkedList<DataEntry>();
		for(int i = 0; i < x.length; i++){
			this.dataSet.add(new DataEntry(x[i], y[i]));
		}
	}
	
	public void setWeighing(DistanceWeighing w){
		this.weight = w;
	}
	
	public static enum DistanceWeighing{
		INDICATOR, INVERSE
	}

	@Override
	public Map<Double, Double> probability(double[] x) {
		HashMap<Double,Double> classcount = new HashMap<Double,Double>();
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