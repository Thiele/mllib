package nu.thiele.ailib.regression;

import java.util.List;

import Jama.Matrix;
import nu.thiele.ailib.data.Data.DataEntry;

public class RadialRegressor implements IRegressor{
	private List<DataEntry> trainingset;
	private BasisFunction[] functions;
	private double[] weights;
	private double regularizationParameter = 0; //Unless set, use 0
	public RadialRegressor(List<DataEntry> trainingset, BasisFunction[] functions, double regParam){
		if(trainingset == null || trainingset.size() == 0 || functions == null || trainingset.get(0).getX().length != functions.length){
			throw new IllegalArgumentException();
		}
		this.trainingset = trainingset;
		this.regularizationParameter = regParam;
		this.functions = functions;
		this.calculateWeights();
	}
	public RadialRegressor(List<DataEntry> trainingset, BasisFunction[] functions){
		if(trainingset == null || trainingset.size() == 0 || functions == null || trainingset.get(0).getX().length != functions.length){
			throw new IllegalArgumentException();
		}
		this.trainingset = trainingset;
		this.functions = functions;
		this.calculateWeights();
	}
	
	private void calculateWeights(){
		//Make target vector
				Matrix targets = new Matrix(trainingset.size(),1);
				int i = 0;
				for(DataEntry d : trainingset){
					targets.set(i, 0, Double.valueOf(d.getY().toString()));
					i++;
				}
				targets = targets.transpose();
				
				//Make design matrix
				Matrix designMatrix = new Matrix(trainingset.size(), functions.length+1);
				i = 0;
				for(DataEntry d : trainingset){
					designMatrix.set(i, 0, 1); //For bias
					for(int j = 1; j < functions.length+1; j++){
						double val = functions[j-1].compute(d.getX()[j-1]);
						designMatrix.set(i, j, val);
					}
					i++;
				}
				Matrix dInv = designMatrix.transpose().times(designMatrix).plus(Matrix.identity(designMatrix.getColumnDimension(), designMatrix.getColumnDimension()).times(this.regularizationParameter));
				dInv = dInv.inverse();
				dInv = dInv.times(designMatrix.transpose());
				dInv = dInv.times(targets.transpose());
				
				this.weights = new double[dInv.transpose().getColumnDimension()];
				for(i = 0; i < this.weights.length; i++){
					this.weights[i] = dInv.get(i, 0);
				}
	}
	
	@Override
	public double regress(double[] x) {
		double retval = this.weights[0]; //Start with bias
		for(int i = 1; i < this.weights.length; i++){
			retval += this.functions[i-1].compute(x[i-1])*this.weights[i];
		}
		return retval;
	}
	
	public double[] getParameters(){
		return this.weights;
	}
	
	public static interface BasisFunction{
		public double compute(double x);
	}
}
