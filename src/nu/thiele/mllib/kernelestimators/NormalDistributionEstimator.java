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
	protected double estimatedProbability(double[] parameters, double x) {
		return Statistics.normalDistributionProbability(x, parameters[0], parameters[1]*parameters[1]);
	}

	@Override
	protected double[] getEstimatorParametersForEstimation() {
		double avg = Statistics.mean(this.ar);
		double std = Statistics.standardDeviation(this.ar,avg);
		double[] retval = {avg, std}; //Avg and standard deviation
		return retval;
	}
}