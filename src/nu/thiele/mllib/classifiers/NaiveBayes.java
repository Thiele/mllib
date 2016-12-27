package nu.thiele.mllib.classifiers;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import nu.thiele.mllib.data.Data.DataEntry;
import nu.thiele.mllib.kernelestimators.GaussianKernelEstimator;
import nu.thiele.mllib.kernelestimators.KernelEstimator;
import nu.thiele.mllib.kernelestimators.KernelEstimatorFactory;
import nu.thiele.mllib.utils.Statistics;
import nu.thiele.mllib.utils.Utils;

public class NaiveBayes implements IClassifier {
	private List<DataEntry> data; 
	private HashMap<Double, Double> classCount;
	private HashMap<Double, HashMap<Integer,KernelEstimator>> ke;
	private KernelEstimatorFactory estFact;
	public NaiveBayes(){
		this(null);
	}
	
	public NaiveBayes(KernelEstimatorFactory fact){
		if(estFact == null) this.estFact = new GaussianKernelEstimator();
		else this.estFact = fact;
		this.init();
	}
	
	private void init(){
		this.classCount = new HashMap<Double,Double>();
		this.ke = new HashMap<Double, HashMap<Integer, KernelEstimator>>();
	}
	
	private void addValue(DataEntry t){
		this.data.add(t);
		if(!this.classCount.containsKey(t.getY())) this.classCount.put(t.getY(), 1.0);
		else this.classCount.put(t.getY(), this.classCount.get(t.getY())+1.0);
		
		for(int i = 0; i < t.getX().length; i++){
			if(this.ke.get(t.getY()) == null) this.ke.put(t.getY(), new HashMap<Integer, KernelEstimator> ());
			KernelEstimator ketemp = this.ke.get(t.getY()).get(i);
			if(ketemp == null){
				ketemp = this.estFact.newInstance();
			}
			ketemp.addValue(t.getX()[i]);
			this.ke.get(t.getY()).put(i, ketemp);
		}

	}
	
	public double classify(double[] x){
		Map<Double, Double> probs = this.probability(x);
		double mProb = Double.MIN_VALUE;
		Double mClass = -1.0;
		for(Double c : probs.keySet()){
			if(probs.get(c) > mProb){
				mProb = probs.get(c);
				mClass = c;
			}
		}
		return mClass;
	}
	
	private double probability(double x, double c, int index){
		List<Double> thisclass = new LinkedList<Double>();
		for(DataEntry d : this.data){
			if(d.getY() == c) thisclass.add(d.getX()[index]);
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
	public void train(double[][] x, double[] y) {
		this.data = new LinkedList<DataEntry>();
		for(int i = 0; i < x.length; i++){
			addValue(new DataEntry(x[i], y[i]));
		}
	}

	@Override
	public Map<Double, Double> probability(double[] x) {
		Map<Double, Double> probs = new HashMap<Double, Double>();
		for(int i = 0; i < x.length; i++){
			for(Double c : this.classCount.keySet()){
				double prob = this.probability(x[i], c, i);
				if(!probs.containsKey(c)) probs.put(c, prob);
				else probs.put(c, probs.get(c)+prob);
			}
		}
		for(int i = 0; i < x.length; i++){
			for(Double o : probs.keySet()){
				double estimatedProb = this.ke.get(o).get(i).probability(x[i]);
				probs.put(o, estimatedProb * probs.get(o));
			}
		}
		Utils.percentify(probs);
		return probs;
	}
}
