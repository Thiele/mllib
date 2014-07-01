package nu.thiele.mllib.classifiers;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import Jama.Matrix;
import nu.thiele.mllib.data.Data.DataEntry;
import nu.thiele.mllib.exceptions.InvalidArgumentException;
import nu.thiele.mllib.utils.Utils;

public class LinearDiscriminantAnalysis implements IClassifier{
	private double[] totalMean;
	private HashMap<Object,Double> classCounts;
	private HashMap<Object,double[]> means;
	private HashMap<Object,Matrix> covariances;
	private List<DataEntry> data;
	private Matrix inverseWithinGroupCovariation;
	
	public LinearDiscriminantAnalysis(List<DataEntry> d) throws InvalidArgumentException{
		this.setTrainingData(d);
		this.loadClassifier();
	}
	
	@Override
	public Object classify(double[] x) {
		Map<Object,Double> sums = this.calculateProbabilityForClassifications(x);
		Object bestObject = null;
		double bestFunction = Double.MIN_VALUE;
		for(Object o : this.covariances.keySet()){
			if(sums.get(o) > bestFunction){
				bestFunction = sums.get(o);
				bestObject = o;
			}
		}
		return bestObject;
	}

	@Override
	public void loadClassifier() {
		//Calculate class counts
		this.classCounts = new HashMap<Object,Double>();
		for(DataEntry d : this.data){
			if(this.classCounts.containsKey(d.getY())){
				this.classCounts.put(d.getY(), this.classCounts.get(d.getY())+1);
			}
			else this.classCounts.put(d.getY(), 1.0);	
		}
		
		//Calculate means
		this.totalMean = new double[this.data.get(0).getX().length];
		this.means = new HashMap<Object,double[]>();
		for(DataEntry d : this.data){
			double[] val = this.means.get(d.getY());
			if(val == null){
				val = new double[d.getX().length];
			}
			for(int i = 0; i < val.length; i++){
				this.totalMean[i] += d.getX()[i];
				val[i] += d.getX()[i];
			}
			this.means.put(d.getY(), val);
		}
		for(Object o : this.means.keySet()){
			double[] val = this.means.get(o);
			for(int i = 0; i < val.length; i++){
				val[i] = val[i]/(this.classCounts.get(o));
			}
		}
		for(int i = 0; i < this.totalMean.length; i++){
			this.totalMean[i] = this.totalMean[i]/((double)this.data.size());
		}
		
		//Calculate covariances
		this.covariances = new HashMap<Object,Matrix>();
		for(DataEntry d : this.data){
			Matrix cov = this.covariances.get(d.getY());
			if(cov == null){
				cov = new Matrix(d.getX().length, d.getX().length); //will be square matrix
				this.covariances.put(d.getY(), cov);
			}
			double[][] vector = new double[1][];
			double[][] meanVector = new double[1][];
			vector[0] = d.getX();
			meanVector[0] = this.means.get(d.getY());
			Matrix v = new Matrix(vector).minus(new Matrix(meanVector));
			Matrix transpose = v.transpose();
			Matrix product = transpose.times(v);
			cov = cov.plus(product);
			this.covariances.put(d.getY(), cov);
		}
		//Now, correct the estimate
		for(Object o : this.covariances.keySet()){			
			Matrix m = this.covariances.get(o);
			m = m.times(1.0/this.classCounts.get(o));
			this.covariances.put(o, m);
		}
		
		//Calculate pooled within group matrix
		this.inverseWithinGroupCovariation = new Matrix(this.totalMean.length, this.totalMean.length);
		for(Object o : this.covariances.keySet()){
			this.inverseWithinGroupCovariation = this.inverseWithinGroupCovariation.plus(this.covariances.get(o).times(this.classCounts.get(o)));
		}
		this.inverseWithinGroupCovariation = this.inverseWithinGroupCovariation.times(1.0/((double)this.data.size()));
		//Invert
		this.inverseWithinGroupCovariation = this.inverseWithinGroupCovariation.inverse();
	}

	@Override
	public void setTrainingData(List<DataEntry> d) throws InvalidArgumentException {
		if(d == null) throw new InvalidArgumentException("Null argument");
		if(d.size() == 0) throw new InvalidArgumentException("Trainingset has no data");
		this.data = d;
	}

	@Override
	public Map<Object, Double> calculateProbabilityForClassifications(double[] x) {
		Map<Object,Double> retval = new HashMap<Object,Double>();
		Matrix xMatrix = new Matrix(new double[][]{x});
		for(Object o : this.covariances.keySet()){ //Find best fitting discriminant function
			Matrix meanVector = new Matrix(new double[][]{this.means.get(o)});
			double a = meanVector.times(this.inverseWithinGroupCovariation).times(xMatrix.transpose()).getArray()[0][0];
			double b = meanVector.times(0.5).times(this.inverseWithinGroupCovariation).times(meanVector.transpose()).getArray()[0][0];
			
			double sum = a-b;
			sum += Math.log(((double)this.data.size())/this.classCounts.get(o));
			retval.put(o, sum);
		}
		Utils.percentify(retval);
		return retval;
	}
}