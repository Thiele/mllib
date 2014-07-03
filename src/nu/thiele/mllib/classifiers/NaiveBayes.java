package nu.thiele.mllib.classifiers;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import nu.thiele.mllib.data.Data.DataEntry;
import nu.thiele.mllib.exceptions.InvalidArgumentException;
import nu.thiele.mllib.kernelestimators.CosineKernelEstimator;
import nu.thiele.mllib.kernelestimators.EpanechnikovKernelEstimator;
import nu.thiele.mllib.kernelestimators.GaussianKernelEstimator;
import nu.thiele.mllib.kernelestimators.HistogramEstimator;
import nu.thiele.mllib.kernelestimators.IEstimator;
import nu.thiele.mllib.kernelestimators.IEstimator.Estimator;
import nu.thiele.mllib.kernelestimators.TriweightKernelEstimator;
import nu.thiele.mllib.utils.Statistics;
import nu.thiele.mllib.utils.Utils;

public class NaiveBayes implements IClassifier {
	private List<DataEntry> data; 
	private HashMap<Object, Double> classCount;
	private HashMap<Object, HashMap<Integer,IEstimator>> ke;
	private Estimator est;
	public NaiveBayes(List<DataEntry> set) throws Exception{
		this.init();
		this.setTrainingData(set);
		this.loadClassifier();
	}
	
	public NaiveBayes(List<DataEntry> set, Estimator estimator) throws Exception{
		this.init();
		this.est = estimator;
		this.setTrainingData(set);
		this.loadClassifier();
	}
	
	private void init(){
		this.classCount = new HashMap<Object,Double>();
		this.ke = new HashMap<Object, HashMap<Integer, IEstimator>>();
	}
	
	private void addValue(DataEntry t){
		if(!this.classCount.containsKey(t.getY())) this.classCount.put(t.getY(), 1.0);
		else this.classCount.put(t.getY(), this.classCount.get(t.getY())+1.0);
		
		for(int i = 0; i < t.getX().length; i++){
			if(this.ke.get(t.getY()) == null) this.ke.put(t.getY(), new HashMap<Integer, IEstimator> ());
			IEstimator ketemp = this.ke.get(t.getY()).get(i);
			if(ketemp == null){
				if(this.est == null) ketemp = new NormalDistributionEstimator();
				else{
					switch(this.est){
					case COSINE:
						ketemp = new CosineKernelEstimator();
						break;
					case EPANECHNIKOV:
						ketemp = new EpanechnikovKernelEstimator();
						break;
					case GAUSSIAN:
						ketemp = new GaussianKernelEstimator();
						break;
					case HISTOGRAM:
						ketemp = new HistogramEstimator();
						break;
					case TRIWEIGHT:
						ketemp = new TriweightKernelEstimator();
						break;
					default:
						ketemp = new NormalDistributionEstimator();
						break;
					}
				}
			}
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
	
	private class NormalDistributionEstimator implements IEstimator{
		private ArrayList<Double> ar =new ArrayList<Double>();
		
		public void addValue(double x){
			this.ar.add(x);
		}
		

		public double probability(double x) {
			double gennemsnit = Statistics.mean(this.ar);
			double std = Statistics.standardDeviation(this.ar, gennemsnit);
			return Statistics.normalDistributionProbability(x, gennemsnit, std*std);
		}
	}
}
