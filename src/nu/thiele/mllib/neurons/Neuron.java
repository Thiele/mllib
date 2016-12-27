package nu.thiele.mllib.neurons;

import java.util.HashMap;
import java.util.LinkedList;

public abstract class Neuron {
	public LinkedList<Neuron> forwardConnections = new LinkedList<Neuron>();;
	protected LinkedList<Neuron> backwardConnections = new LinkedList<Neuron>();
	protected int numInputsReceived = 0;
	protected double sumOfInputs = 0;
	protected double latestOutput = 0;
	protected HashMap<Neuron, Double> weights = new HashMap<Neuron, Double>();
	protected HashMap<Neuron, Double> tmpWeights = new HashMap<Neuron, Double>();
	private String identifier;
	private double error;
		
	public Neuron(String id){
		identifier = id;
	}
	
	public String toString(){
		return this.identifier;
	}
		
	void addConnectionBackward(Neuron n){
		backwardConnections.add(n);
	}
	
	public void addConnectionForward(Neuron n, double weight){
		forwardConnections.add(n);
		weights.put(n, weight);
		n.addConnectionBackward(this);
	}
	
	public LinkedList<Neuron> getBackwardsConnections(){
		return this.backwardConnections;
	}
	
	public LinkedList<Neuron> getFowrardsConnections(){
		return this.forwardConnections;
	}
		
	public String getIdentifier(){
		return identifier;
	}
		
	public double getLatestOutput(){
		return latestOutput;
	}
	
	public double getWeight(Neuron n){
		return this.weights.get(n);
	}
	
	public void setWeight(Neuron n, double w){
		this.tmpWeights.put(n, w);
	}
	
	public void commitWeights(){
		for(Neuron n : this.tmpWeights.keySet()) this.weights.put(n, this.tmpWeights.get(n));
	}
		
	public void input(double d){
		sumOfInputs += d;
		numInputsReceived++;
		
		if(numInputsReceived >= backwardConnections.size()) fire();
	}
		
	public void fire(){
		latestOutput = activationFunction(sumOfInputs);
		for(Neuron n : forwardConnections) n.input(latestOutput*weights.get(n));
	}
		
	public void reset(){
		this.error = 0;
		sumOfInputs = 0;
		numInputsReceived = 0;
		this.tmpWeights.clear();
	}
	
	public static double activationFunction(double d){
		return 1.0/(1+Math.pow(Math.E, -d));
	}
	
	public static double activationFunctionDerivative(double d){
		return (1-d);
	}

	public double getError() {
		return error;
	}

	public void setError(double error) {
		this.error = error;
	}
}
