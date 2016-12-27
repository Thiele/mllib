package nu.thiele.mllib.classifiers;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;
import nu.thiele.mllib.neurons.HiddenNeuron;
import nu.thiele.mllib.neurons.InputNeuron;
import nu.thiele.mllib.neurons.Neuron;
import nu.thiele.mllib.neurons.OutputNeuron;
import nu.thiele.mllib.regression.IRegressor;

/**
 * @author Andreas Thiele
 */
public class MultiLayerPerceptron implements IClassifier, IMultiClassifier{
	/**
	 * Example of the network solving the XOR-problem  
	 */
	public static void main(String[] args){
		MultiLayerPerceptron mlp = new MultiLayerPerceptron(1);
		
		//mlp.build(2,new int[]{2}, 3);
		
		double[][] x = new double[4][];
		x[0] = new double[]{0.0,0.0};
		x[1] = new double[]{0.0,1.0};
		x[2] = new double[]{1.0,0.0};
		x[3] = new double[]{1.0,1.0};
		double[] y = {7,3,3,1};

		mlp.train(x, y, 5000);
		
		for(int i = 0; i < 4; i++){
			System.out.println("Classifying: "+i+", y: "+mlp.classify(x[i])+", real value: "+y[i]);
		}
	}
	
	private ArrayList<InputNeuron> inputs = new ArrayList<InputNeuron>();
	private ArrayList<ArrayList<Neuron>> hiddenLayers = new ArrayList<ArrayList<Neuron>>();
	private ArrayList<OutputNeuron> outputs = new ArrayList<OutputNeuron>();
	private ArrayList<InputNeuron> biases = new ArrayList<InputNeuron>();
	private HashMap<String, Neuron> idToNeuronMap = new HashMap<String,Neuron>();
	private TreeMap<Double, Integer> classToOutputIndexMap = new TreeMap<Double,Integer>();
	private TreeMap<Integer, Double> outputIndexToClassMap = new TreeMap<Integer, Double>();
	private Random rand;
	private double learningRate;
	private boolean isBuilt;
	
	public MultiLayerPerceptron(int numHidden){
		this(numHidden, 0.5);
	}
	
	public MultiLayerPerceptron(int numHidden, double lr){
		this.learningRate = lr;
		for(int i = 0; i <= numHidden; i++){
			InputNeuron bias;
			if(i == 0){
				bias = new InputNeuron("B I");
			}
			else{
				hiddenLayers.add(new ArrayList<Neuron>());
				bias = new InputNeuron("B H "+i);
			}
			biases.add(bias);
			this.idToNeuronMap.put(bias.getIdentifier(), bias);
		}
	}
	
	public void printNetwork(){
		System.out.println("===== INPUTS =====");
		for(InputNeuron in : this.inputs){
			System.out.println("=== "+in.getIdentifier());
			for(Neuron n : in.getFowrardsConnections()){
				System.out.println("=> "+n.getIdentifier()+" weight: "+in.getWeight(n));
			}
		}
		System.out.println("===== HIDDEN LAYERS =====");
		for(int i = 0; i < this.hiddenLayers.size(); i++){
			System.out.println("=== Layer "+(i+1));
			for(Neuron n : this.hiddenLayers.get(i)){
				System.out.println("=== "+n.getIdentifier());
				for(Neuron next : n.getFowrardsConnections()){
					System.out.println("=> "+next.getIdentifier()+" weight: "+n.getWeight(next));					
				}
			}
		}
		System.out.println("===== BIAS =====");
		for(InputNeuron in : this.biases){
			System.out.println("=== "+in.getIdentifier());
			for(Neuron n : in.getFowrardsConnections()){
				System.out.println("=> "+n.getIdentifier()+" weight: "+in.getWeight(n));
			}
		}
	}
	
	private double getRandomWeight(){
		if(this.rand == null) this.rand = new Random();
		return this.rand.nextDouble() * (this.rand.nextBoolean() ? 1 : - 1);
	}
	
	public void build(int input, int[] hiddens, int output){
		//Build only if not done so already
		if(this.isBuilt) return;
		
		//Add the neurons
		for(int i = 1; i <= input; i++) this.addInput();
		for(int i = 1; i <= output; i++) this.addOutput();
		//Is something set for hiddens? Otherwise just use 10
		if(hiddens.length == 0) hiddens = new int[]{10};
		
		for(int h = 0; h < hiddens.length; h++){
			for(int i = 0; i < hiddens[h]; i++) this.addHidden(h);
		}
		
		//Add biases first
		for(int i = 0; i < biases.size(); i++){
			if(i == biases.size()-1){
				for(OutputNeuron o : outputs) biases.get(i).addConnectionForward(o, this.getRandomWeight());
			}
			else{
				for(Neuron n : hiddenLayers.get(i)) biases.get(i).addConnectionForward(n, this.getRandomWeight());
			}
		}
		for(InputNeuron in : this.inputs){
			//No hidden layers. Just connect to output
			if(this.hiddenLayers.size() == 0){
				for(OutputNeuron o : this.outputs) in.addConnectionForward(o, this.getRandomWeight());
			}
			else{
				//There are hidden layers. Add connections to first
				for(Neuron n : this.hiddenLayers.get(0)) in.addConnectionForward(n, this.getRandomWeight());
			}
		}
		
		
		//If any hidden layers, add layers moving forwards for all
		for(int i = 0; i < hiddenLayers.size(); i++){
			for(Neuron n : hiddenLayers.get(i)){
				if(i == hiddenLayers.size()-1){
					for(OutputNeuron o : this.outputs) n.addConnectionForward(o, this.getRandomWeight());
				}
				else{
					for(Neuron next : this.hiddenLayers.get(i+1)) n.addConnectionForward(next, this.getRandomWeight());
				}
			}
		}
		this.isBuilt = true;
	}
	
	private void addInput(){
		InputNeuron i = new InputNeuron("I "+(this.inputs.size()+1));
		this.inputs.add(i);
		this.idToNeuronMap.put(i.getIdentifier(), i);
	}
	
	private void addHidden(int numLayer){
		HiddenNeuron n = new HiddenNeuron("H "+(numLayer+1)+" "+(this.hiddenLayers.get(numLayer).size()+1));
		this.hiddenLayers.get(numLayer).add(n);
		this.idToNeuronMap.put(n.getIdentifier(), n);
	}
	
	private void addOutput(){
		OutputNeuron o = new OutputNeuron("O "+(this.outputs.size()+1));
		this.outputs.add(o);
		this.idToNeuronMap.put(o.getIdentifier(), o);
	}
	
	public void saveNetwork(String path) throws IOException{
		PrintWriter writer = new PrintWriter(path, "UTF-8");
		HashSet<String> allIds = new HashSet<String>();
		//Save nodes
		writer.println("I "+this.inputs.size());
		writer.println("O "+this.outputs.size());
		for(Neuron n : this.inputs){
			allIds.add(n.getIdentifier());
		}
		int i = 1;
		for(List<Neuron> list : this.hiddenLayers){
			writer.println("H "+i+" "+list.size());
			i++;
			for(Neuron n : list){
				allIds.add(n.getIdentifier());
			}
		}
		for(Neuron n : this.outputs){
			allIds.add(n.getIdentifier());
		}
		//Save biases
		for(Neuron n : this.biases){
			allIds.add(n.getIdentifier());
		}
		//Save all weights
		for(String id : allIds){
			Neuron from = this.idToNeuronMap.get(id);
			for(Neuron to : from.forwardConnections){
				writer.println("W "+from.getIdentifier()+","+to.getIdentifier()+","+from.getWeight(to));				
			}
		}
		
		writer.close();
	}
	
	@Override
	public double[] classifyMultipleOutputs(double[] input){
		//Reset anything first
		for(Neuron n : this.idToNeuronMap.values()) n.reset();
		
		//Fire biases first
		for(InputNeuron b : this.biases) b.input(1);
		for(int i = 0; i < input.length; i++) this.inputs.get(i).input(input[i]);
		
		double[] retval = new double[this.outputs.size()];
		for(int i = 0; i < this.outputs.size(); i++) retval[i] = this.outputs.get(i).getLatestOutput();
		return retval;
	}
	
	private double backpropagate(double[] guesses, double[] targets){
		double totalError = this.getTotalErrorSingle(guesses, targets);
		//Don't worry about biases. They will be handled fine without explicitly using them
		//For last hidden => output layer
		for(int i = 0; i < this.outputs.size(); i++){
			OutputNeuron n = this.outputs.get(i);
			n.setError(guesses[i]*(targets[i]-guesses[i])*Neuron.activationFunctionDerivative(guesses[i]));
			for(Neuron prev : n.getBackwardsConnections()){
				double change = this.learningRate * n.getError() * prev.getLatestOutput();
				prev.setWeight(n, prev.getWeight(n)+change);
			}
		}
		//For all other layers. Note: Look only in hidden layers, since input layer will be handled
		for(int i = this.hiddenLayers.size()-1; i >= 0; i--){
			List<Neuron> layer = this.hiddenLayers.get(i);
			for(Neuron n : layer){
				double errOut = 0;
				for(Neuron next : n.getFowrardsConnections()){
					double tmp = next.getError() * n.getWeight(next);
					errOut += tmp;
				}
				n.setError(errOut*n.getLatestOutput()*Neuron.activationFunctionDerivative(n.getLatestOutput()));
				//And update weights
				for(Neuron prev : n.getBackwardsConnections()){
					double change = this.learningRate * n.getError() * prev.getLatestOutput();
					prev.setWeight(n, prev.getWeight(n)+change);
				}
			}
		}
		
		//And commit weight changes
		this.commitWeightChanges();
		
		return totalError;
	}
	
	private void commitWeightChanges(){
		for(Neuron n : this.idToNeuronMap.values()) n.commitWeights();
	}
	
	public MultiLayerPerceptron copy(){
		MultiLayerPerceptron retval = new MultiLayerPerceptron(this.hiddenLayers.size(), this.learningRate);
		int[] hiddensSizes = new int[]{this.hiddenLayers.size()};
		for(int i = 0; i < this.hiddenLayers.size(); i++){
			hiddensSizes[i] = this.hiddenLayers.get(i).size();
		}
		retval.build(this.inputs.size(), hiddensSizes, this.outputs.size());
		for(String id : this.idToNeuronMap.keySet()){
			for(Neuron n : this.idToNeuronMap.get(id).forwardConnections){
				retval.setWeight(id, n.getIdentifier(), this.idToNeuronMap.get(id).getWeight(n));
			}
		}
		return retval;
	}
	
	public double getTotalError(double[][] guesses, double[][] targets){
		double sum = 0;
		for(int i = 0; i < guesses.length; i++) sum += this.getTotalErrorSingle(guesses[i], targets[i]);
		return sum;
	}
	
	public double getTotalErrorSingle(double[] guesses, double[] targets){
		double totalError = 0;
		for(int i = 0; i < guesses.length; i++){
			double err = guesses[i]-targets[i];
			totalError += err*err/2.0;
		}
		return totalError;
	}
	
	@Override
	public void train(double[][] xs, double[] ys){
		this.train(xs, ys, 1000);
	}
	
	public void train(double[][] xs, double[] ys, int times){
		if(this.classToOutputIndexMap.isEmpty()){
			for(double y : ys){
				if(!this.classToOutputIndexMap.containsKey(y)){
					this.classToOutputIndexMap.put(y, this.classToOutputIndexMap.size());
					this.outputIndexToClassMap.put(this.classToOutputIndexMap.get(y), y);
				}
			}
		}
		
		if(!this.isBuilt){
			//Count different classes
			HashSet<Double> classes = new HashSet<Double>();
			for(double d : ys){
				if(!classes.contains(d)) classes.add(d);
			}
			this.build(xs[0].length, new int[]{}, classes.size());
		}
		
		for(int i = 1; i <= times; i++){
			for(int j = 0; j < xs.length; j++){
				double[] guesses = this.classifyMultipleOutputs(xs[j]);
				this.backpropagate(guesses, this.oneHotEncode(this.classToOutputIndexMap.get(ys[j])));
			}
		}
	}
	
	private double[] oneHotEncode(double val){
		double[] retval = new double[this.outputs.size()];
		for(double d = 0; d < retval.length; d++){
			if(d == val) retval[(int) d] = 1;
		}
		return retval;
	}
	
	
	public void setWeightRandom(String fromId, String toId){
		this.setWeightRandom(this.idToNeuronMap.get(fromId), this.idToNeuronMap.get(toId));
	}

	private void setWeightRandom(Neuron from, Neuron to){
		from.setWeight(to, this.rand.nextDouble());	
		from.commitWeights();
	}
	
	private void setWeight(Neuron from, Neuron to, double d){
		from.setWeight(to, d);
		from.commitWeights();
	}
	
	public void setWeight(String fromId, String toId, double d){
		this.setWeight(this.idToNeuronMap.get(fromId), this.idToNeuronMap.get(toId), d);
	}
	
	public static MultiLayerPerceptron loadPerceptron(String path) throws IOException{
		BufferedReader br = new BufferedReader(new FileReader(path));

	    int numInputs = 0;
	    int numOutputs = 0;
	    List<String> weights = new LinkedList<String>();
	    TreeMap<Integer,Integer> hiddenLayerSizes = new TreeMap<Integer,Integer>();
	    String line;
	    while ((line = br.readLine()) != null) {
	    	if(line.startsWith("H ")){
	    		String[] split = line.split(" ");
	    		hiddenLayerSizes.put(Integer.parseInt(split[1]), Integer.parseInt(split[2]));
	    	}
	    	else if(line.startsWith("I ")) numInputs = Integer.parseInt(line.substring(2)+"");
	    	else if(line.startsWith("O ")) numOutputs = Integer.parseInt(line.substring(2)+"");
	    	else if(line.startsWith("W ")) weights.add(line.substring(2));
	    }
	    br.close();
	    MultiLayerPerceptron retval = new MultiLayerPerceptron(hiddenLayerSizes.size());
	    
	    //Now actually build network
	    int[] hiddens = new int[hiddenLayerSizes.size()];
	    for(int i = 0; i < hiddenLayerSizes.size(); i++) hiddens[i] = hiddenLayerSizes.get(i);
	    retval.build(numInputs, hiddens, numOutputs);
	    
	    //Connections are now added. Set the weights up
	    for(String l : weights){
	    	String[] split = l.split(",");
	    	double weight = Double.parseDouble(split[2]);
	    	retval.setWeight(split[0],split[1], weight);
	    }
	    
		return retval;
	}

	@Override
	public double classify(double[] x) {
		double[] guess = this.classifyMultipleOutputs(x);
		int maxIndex = 0;
		double max = -1;
		for(int i = 0; i < this.outputs.size(); i++){
			if(guess[i] > max){
				max = guess[i];
				maxIndex = i;
			}
		}
		return this.outputIndexToClassMap.get(maxIndex);
	}

	@Override
	public Map<Double, Double> probability(double[] x) {
		// TODO Auto-generated method stub
		return null;
	}
}
