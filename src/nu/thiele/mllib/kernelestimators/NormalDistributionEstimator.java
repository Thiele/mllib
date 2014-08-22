package nu.thiele.mllib.kernelestimators;

import java.util.ArrayList;

import nu.thiele.mllib.utils.Statistics;

public class NormalDistributionEstimator extends KernelEstimator{
	private ArrayList<Double> ar =new ArrayList<Double>();
	
	@Override
	protected void addValueToEstimator(double x){
		this.ar.add(x);
	}
	
	@Override
	protected double estimatedProbability(Object[] parameters, double x) {
		return Statistics.normalDistributionProbability(x, (double)parameters[0], ((double)parameters[1])*((double)parameters[1]));
	}

	@Override
	protected Object[] getEstimatorParametersForEstimation() {
		double avg = Statistics.mean(this.ar);
		double std = Statistics.standardDeviation(this.ar,avg);
		Object[] retval = {avg, std}; //Avg and standard deviation
		return retval;
	}
}