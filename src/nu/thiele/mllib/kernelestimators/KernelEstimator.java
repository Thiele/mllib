package nu.thiele.mllib.kernelestimators;
/**
 * For any kernel, see: http://en.wikipedia.org/wiki/Kernel_(statistics)
 * @author Andreas Thiele
 *
 */
public abstract class KernelEstimator {
	/*
	 * Abstract ones to implement
	 */
	protected abstract void addValueToEstimator(double x);
	protected abstract double estimatedProbability(Object[] parameters, double x);
	protected abstract Object[] getEstimatorParametersForEstimation();
	
	/*
	 * Internal variables
	 */
	private boolean changedSinceLastParameterCalculation = true;
	private Object[] estimatorParameters;
	/*
	 * 
	 * Methods in this abstraction
	 * 
	 */
	public void addValue(double x){
		this.addValueToEstimator(x);
		changedSinceLastParameterCalculation = true;
	}
	
	public double probability(double x){
		if(this.changedSinceLastParameterCalculation){
			this.estimatorParameters = this.getEstimatorParametersForEstimation();
			this.changedSinceLastParameterCalculation = false;
		}
		return this.estimatedProbability(this.estimatorParameters, x);
	}
	
	public enum Estimator{
		COSINE, EPANECHNIKOV, GAUSSIAN, HISTOGRAM, TRIWEIGHT
	}
}
