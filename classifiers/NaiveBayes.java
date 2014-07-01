package nu.thiele.ailib.classifiers;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import nu.thiele.ailib.data.Data.DataEntry;
import nu.thiele.ailib.exceptions.InvalidArgumentException;
import nu.thiele.ailib.utils.Statistics;
import nu.thiele.ailib.utils.Utils;

public class NaiveBayes implements IClassifier {
	private List<DataEntry> data; 
	private HashMap<Object, Double> classCount;
	private HashMap<Object, HashMap<Integer,NormalDistributionEstimator>> ke;
	public NaiveBayes(List<DataEntry> set) throws Exception{
		//Init
		this.classCount = new HashMap<Object,Double>();
		this.ke = new HashMap<Object, HashMap<Integer, NormalDistributionEstimator>>();
		
		this.setTrainingData(set);
		this.loadClassifier();
	}
	
	private void addValue(DataEntry t){
		if(!this.classCount.containsKey(t.getY())) this.classCount.put(t.getY(), 1.0);
		else this.classCount.put(t.getY(), this.classCount.get(t.getY())+1.0);
		
		for(int i = 0; i < t.getX().length; i++){
			if(this.ke.get(t.getY()) == null) this.ke.put(t.getY(), new HashMap<Integer, NormalDistributionEstimator> ());
			NormalDistributionEstimator ketemp = this.ke.get(t.getY()).get(i);
			if(ketemp == null) ketemp = new NormalDistributionEstimator();
			ketemp.addValue(t.getX()[i]);
			this.ke.get(t.getY()).put(i, ketemp);
		}

	}
	
	public Object classify(double[] x){
		Map<Object, Double> probs = this.calculateProbabilityForClassifications(x);
		double mProb = Double.MIN_VALUE;
		Object mClass = null;
		for(Object c : probs.keySet()){
			if(probs.get(c) > mProb){
				mProb = probs.get(c);
				mClass = c;
			}
		}
		return mClass;
	}
	
	private double probability(double x, Object c, int index){
		List<Double> thisclass = new LinkedList<Double>();
		for(DataEntry d : this.data){
			if(d.getY().getClass().equals(c.getClass())) thisclass.add(d.getX()[index]);
		}
		double p = this.probability(x, thisclass);
		return p;
	}
	
	private double probability(double x, List<Double> datas) {
		double mean = Statistics.mean(datas);
		double std = Statistics.standardDeviation(datas, mean);
		double exp = -(x-mean)*(x-mean)/(2*std*std);
		double base = 1.0/(Math.sqrt(2*Math.PI)*std);
		double retur = base*Math.pow(Math.E, exp);
		return retur;
	}

	@Override
	public void loadClassifier() {
		for(DataEntry tse : data){
			addValue(tse);
		}
	}

	@Override
	public void setTrainingData(List<DataEntry> data)
			throws InvalidArgumentException {
		this.data = data;
	}

	@Override
	public Map<Object, Double> calculateProbabilityForClassifications(double[] x) {
		Map<Object, Double> probs = new HashMap<Object, Double>();
		for(int i = 0; i < x.length; i++){
			for(Object c : this.classCount.keySet()){
				double prob = this.probability(x[i], c, i);
				if(!probs.containsKey(c)) probs.put(c, prob);
				else probs.put(c, probs.get(c)+prob);
			}
		}
		for(int i = 0; i < x.length; i++){
			for(Object o : probs.keySet()){
				double estimatedProb = this.ke.get(o).get(i).probability(x[i]);
				probs.put(o, estimatedProb * probs.get(o));
			}
		}
		Utils.percentify(probs);
		return probs;
	}
	
	private class NormalDistributionEstimator{
		private ArrayList<Double> ar;
		public NormalDistributionEstimator(){
			ar = new ArrayList<Double>();
		}
		
		public void addValue(double x){
			this.ar.add(x);
		}
		

		public double probability(double x) {
			double gennemsnit = Statistics.mean(this.ar);
			double std = Statistics.standardDeviation(this.ar, gennemsnit);
			double exp = -(x-gennemsnit)*(x-gennemsnit)/(2*std*std);
			double base = 1.0/(Math.sqrt(2*Math.PI)*std);
			double retur = base*Math.pow(Math.E, exp);
			return retur;
		}
	}
}