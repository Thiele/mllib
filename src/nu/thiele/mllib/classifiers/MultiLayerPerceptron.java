package nu.thiele.mllib.classifiers;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.lang.Math;
import java.util.LinkedList;
import java.util.List;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import nu.thiele.mllib.data.Data.DataEntry;
import nu.thiele.mllib.exceptions.InvalidArgumentException;
import nu.thiele.mllib.regression.IRegressor;
import nu.thiele.mllib.utils.Statistics;
import nu.thiele.mllib.utils.Testing;

/**
 * Implements backpropagation as described in:
 * http://www4.rgu.ac.uk/files/chapter3%20-%20bp.pdf
 * @author Andreas Thiele
 */
public class MultiLayerPerceptron implements IClassifier, IRegressor{
	/**
	 * Example of the network solving the XOR-problem
	 * @throws UnsupportedEncodingException 
	 * @throws FileNotFoundException 
	 */
	public static void main(String[] args) throws FileNotFoundException, UnsupportedEncodingException{
		ActivationFunctions functions = new ActivationFunctions(){
			@Override
			public double activation(double a) {
				return a/(1+Math.abs(a));
			}
			@Override
			public double sigmoid(double d) { //Applied to output neurons to map into wanted space.
				return d;
			}
			@Override
			public double derivative(double a) {
				double n = (1+Math.abs(a));
				return 1/(n*n);
			}
		};
		MultiLayerPerceptron mlp = new MultiLayerPerceptron(2, 2, 	1, 1, 0.1, functions);
		
		double[][] x = new double[4][];
		x[0] = new double[]{0.0,0.0};
		x[1] = new double[]{0.0,1.0};
		x[2] = new double[]{1.0,0.0};
		x[3] = new double[]{1.0,1.0};
		double[] y = {0,1,1,0};
		List<DataEntry> datas = new LinkedList<DataEntry>();
		for(int i = 0; i < x.length; i++){
			datas.add(new DataEntry(x[i], new double[]{y[i]}));
		}

		for(int i = 0; i < 1000000; i++){
			mlp.trainBatch(datas);
		}
		for(int i = 0; i < 4; i++){
			System.out.println("Classifying: "+i+", y: "+mlp.regress(x[i])+", real value: "+y[i]);
		}
	}
	private double learningrate;
	private List<List<Neuron>> hidden;
	private List<Neuron> input;
	private List<Neuron> output;
	private HashMap<Neuron,Integer> inputIndex;
	private HashMap<Integer,HashMap<Neuron,Integer>> hiddenIndex;
	private ActivationFunctions functions;
	private Object[] classes;
	public MultiLayerPerceptron(int input, int hidden, Object[] tClasses, int numberOfHiddenLayers, double learningrate, ActivationFunctions functions){
		this(input,hidden,tClasses.length,numberOfHiddenLayers,learningrate,functions);
		this.classes = tClasses;
	}
	
	
	public MultiLayerPerceptron(int input, int hidden, int output, int numberOfHiddenLayers, double learningrate, ActivationFunctions functions){
		this.hiddenIndex = new HashMap<Integer,HashMap<Neuron,Integer>>();
		this.inputIndex = new HashMap<Neuron,Integer>();
		this.functions = functions;
		this.hidden = new LinkedList<List<Neuron>>();
		this.input = new LinkedList<Neuron>();
 		this.output = new LinkedList<Neuron>();
 		this.learningrate = learningrate;
 		//Input
 		for(int i = 1; i <= input; i++){
 			this.input.add(new Neuron(NeuronType.Input, this.functions, this.learningrate));
 		}
 		for(Neuron i : this.input){
 			this.inputIndex.put(i, this.input.indexOf(i));
 		}
 		
 		//Hidden
 		for(int i = 1; i <= numberOfHiddenLayers; i++){
 			List<Neuron> a = new LinkedList<Neuron>();
 			for(int j = 1; j <= hidden; j++){
 				a.add(new Neuron(NeuronType.Hidden, this.functions, this.learningrate));
 			}
 			this.hidden.add(a);
 		}
 		for(List<Neuron> a : this.hidden){
 			HashMap<Neuron,Integer> put = new HashMap<Neuron,Integer>();
 			for(Neuron h : a){
 				put.put(h, a.indexOf(h));
 			}
 			this.hiddenIndex.put(this.hidden.indexOf(a), put);
 		}
 		
 		//Output
 		for(int i = 1; i <= output; i++){
 			this.output.add(new Neuron(NeuronType.Output, this.functions, this.learningrate));
 		} 		
 		
 		for(Neuron i : this.input){
 			for(Neuron h : this.hidden.get(0)){
 				i.connect(h, Math.random()*(Math.random() > 0.5 ? 1 : -1));
 			}
 		}
 		for(int i = 1; i < this.hidden.size(); i++){
 			for(Neuron h : this.hidden.get(i-1)){
 				for(Neuron hto : this.hidden.get(i)){
 					h.connect(hto, Math.random()*(Math.random() > 0.5 ? 1 : -1));
 				}
 			}
 		}
 		for(Neuron h : this.hidden.get(this.hidden.size()-1)){
 			for(Neuron o : this.output){
 				h.connect(o, Math.random()*(Math.random() > 0.5 ? 1 : -1));
 			}
 		}
 	}
	
	/**
	 * @param exp The expected value of the input
	 */
	private void backpropagate(double[] exp){
		//Calculate output error
		int i = 0;
		for(Neuron outputNeuron : this.output){
			outputNeuron.setError((exp[i] - outputNeuron.getLatestOutput()));
			//And update its bias
			outputNeuron.addBiasChange(outputNeuron.getError());
			i++;
		}
		//From hidden to hidden and input to hidden
		for(i = this.hidden.size()-1; i >= 0; i--){
			for(Neuron h : this.hidden.get(i)){
				double p = this.functions.derivative(h.getLatestSum());
				double k = 0;
				for(Synaps s : h.getconnectedTo()){
					k += s.getTo().getError()*s.getweight();
				}
				h.setError(p * k);
			}
			//Update bias
			for(Neuron neuron : this.hidden.get(i)){
				neuron.addBiasChange(neuron.getError());
			}
		}
		//And update all weights
		for(Neuron n : this.input){
			for(Synaps s : n.getconnectedTo()){
				s.addWeightChange(s.getTo().getError() * n.getLatestInput());
			}
		}
		for(List<Neuron> l : this.hidden){
			for(Neuron n : l){
				for(Synaps s : n.getconnectedTo()){
					s.addWeightChange(s.getTo().getError() * n.getLatestOutput());
				}
			}
		}
	}
	
	/**
	 * 
	 * @param input Input to be classified
	 * @return The classification of the input
	 */
	public Object classify(double[] input) {
		for(int i = 0; i < input.length; i++){
			this.input.get(i).input(input[i]);
		}
		//Propagate stuff
		for(Neuron n : this.input) n.propagate();
		for(List<Neuron> list : this.hidden){
			for(Neuron n : list) n.propagate();
		}
		for(Neuron n : this.output) n.propagate();
		double[] r = new double[this.output.size()];
		int highestIndex = 0;
		double highestVal = Double.MIN_VALUE;
		for(int i = 0; i < r.length; i++){
			r[i] = this.output.get(i).getLatestOutput();
			if(r[i] > highestVal){
				highestIndex = i;
				highestVal = r[i];
			}
		}
		//Find the largest one
		return this.classes[highestIndex];
	}	
	
	private void applyUpdates(){
		for(Neuron n : this.input){
			n.applyBiasChanges();
			for(Synaps s : n.getconnectedTo()) s.applyWeightChange();
		}
		for(List<Neuron> h : this.hidden){
			for(Neuron n : h){
				n.applyBiasChanges();
				for(Synaps s : n.getconnectedTo()) s.applyWeightChange();
			}
		}
		for(Neuron n : this.output){
			n.applyBiasChanges();
			for(Synaps s : n.getconnectedTo()) s.applyWeightChange();
		}
	}
	
	public void train(DataEntry data){
		if(this.classes == null){
			this.regress(data.getX());
		}
		else this.classify(data.getX());
		backpropagate(this.encodeYToDoubleArray(data.getY()));
		this.applyUpdates();
	}
	
	public void trainOnline(List<DataEntry> datas){
		for(DataEntry data : datas){
			this.train(data);
		}
	}
	
	public void train(List<DataEntry> datasOrg, List<DataEntry> testSet, AcceptanceCriteria criteria){
		List<DataEntry> datas = new LinkedList<DataEntry>(datasOrg);
		double mserror = Testing.rootMeanSquareError(this, testSet);
		double trainMsError = Testing.rootMeanSquareError(this, datasOrg);
		for(int epoch = 1; !criteria.isAccepted(epoch, mserror, trainMsError); epoch++){
			Collections.shuffle(datas); //Do not learn some stupid ordering
			this.trainBatch(datas);
			trainMsError = Testing.rootMeanSquareError(this, datas);
			mserror = Testing.rootMeanSquareError(this, testSet);
		}
	}
	
	public void train(List<DataEntry> datasOrg, List<DataEntry> testSet, ClassificationAcceptanceCriteria criteria){
		List<DataEntry> datas = new LinkedList<DataEntry>(datasOrg);
		double acc = Testing.testSet(this, testSet).getAccuracy();
		double tsAcc = Testing.testSet(this, datasOrg).getAccuracy();
		for(int epoch = 1; !criteria.isAccepted(epoch, 1-acc, 1-tsAcc); epoch++){
			Collections.shuffle(datas); //Do not learn some stupid ordering
			this.trainBatch(datas);
			acc = Testing.testSet(this, testSet).getAccuracy();
			tsAcc = Testing.testSet(this, datasOrg).getAccuracy();
		}
	}
	
	public void trainBatch(List<DataEntry> datas){
		for(DataEntry data : datas){
			if(this.classes == null) this.regress(data.getX());
			else this.classify(data.getX());//Forward...
			double[] y = this.encodeYToDoubleArray(data.getY());
			backpropagate(y); //Backward
		}
		this.applyUpdates();
	}
	
	private double[] encodeYToDoubleArray(Object y){
		if(y instanceof double[]) return (double[]) y;
		if(y instanceof Double) return new double[]{Double.valueOf(y.toString())};
		double[] retval = new double[this.classes.length];
		int i = 0;
		for(Object o : this.classes){
			if(o.equals(y)){
				retval[i] = 1;
				break;
			}
			i++;
		}
		return retval;
	}
	
	private class Neuron {
		private double latestoutput = 0, sum, biasWeight = 1, error, learningrate, latestSum, latestInput;
		private List<Synaps> connectedFrom, connectedTo;
		private ActivationFunctions funs;
		private NeuronType type;
		private double biasChange = 0;
		public Neuron(NeuronType type, ActivationFunctions funs, double learningrate){
			this.learningrate = learningrate;
			this.funs = funs;
			this.type = type;
			this.connectedFrom = new LinkedList<Synaps>();
			this.connectedTo = new LinkedList<Synaps>();
		}
		
		private void addFrom(Synaps s){
			this.connectedFrom.add(s);
		}
		
		private void connect(Neuron e, double weight){
			Synaps s = new Synaps(this, e, weight, this.learningrate);
			e.addFrom(s);
			this.connectedTo.add(s);
		}
		
		private double getError(){
			return this.error;
		}
				
		private double getLatestOutput(){
			return this.latestoutput;
		}
		
		private void input(double input){
			this.sum = sum+input;
		}
		
		private void addBiasChange(double d){
			this.biasChange += d;
		}
		
		private void applyBiasChanges(){
			this.biasWeight += this.biasChange * this.learningrate;
			this.biasChange = 0;
		}
		
		private void propagate(){
			this.latestInput = this.sum;
			this.sum += this.biasWeight;
			this.latestSum = this.sum;
			if(this.type == NeuronType.Hidden){
				this.latestoutput = this.funs.activation(this.sum);	
			}
			else if(this.type == NeuronType.Input){
				this.latestoutput = this.sum;
			}
			else if(this.type == NeuronType.Output){
				this.latestoutput = this.funs.sigmoid(this.sum);
			}
			for(Synaps n : this.connectedTo){
				if(this.type == NeuronType.Input || this.type == NeuronType.Hidden){
					n.getTo().input(this.latestoutput*n.getweight());
				}
				//Else output. No need to pass along
			}
			this.sum = 0.0;
		}
		
		private double getLatestSum(){
			return this.latestSum;
		}
		
		private double getLatestInput(){
			return this.latestInput;
		}
		
		private void setError(double e){
			this.error = e;
		}
		
		private List<Synaps> getconnectedTo(){
			return this.connectedTo;
		}
		
		public String toString(){
			String retur = this.hashCode()+" with "+this.connectedTo.size()+" connections";
			return retur;
		}
	}
	
	
	/**
	 * 
	 * Private class to connect neurons
	 *
	 */
	private class Synaps {
		private Neuron to;
		private double weight, weightUpdate = 0, learningrate;
		private Synaps(Neuron from, Neuron to, double weight, double learningrate){
			this.to= to;
			this.weight = weight;
			this.learningrate = learningrate;
		}
		
		private void addWeightChange(double d){
			this.weightUpdate += d;
		}
		private void applyWeightChange(){
			this.weight += this.weightUpdate * this.learningrate;
			this.weightUpdate = 0;
		}
		
		private double getweight(){
			return this.weight;
		}
		
		private Neuron getTo(){
			return this.to;
		}
		
		public String toString(){
			return weight+"";
		}
	}
	
	public static interface AcceptanceCriteria{
		public boolean isAccepted(int epoch, double MSError, double trainMSErorr);
	}
	
	public static interface ClassificationAcceptanceCriteria{
		public boolean isAccepted(int epoch, double error, double trainError);
	}
	
	public static interface ActivationFunctions{
		public double activation(double a);
		public double derivative(double a);
		public double sigmoid(double d);
	}
	public static enum LearningMethod{
		Batch, Online, Stochastic
	}
	private static enum NeuronType{
		Input, Hidden, Output
	}
	@Override
	public void loadClassifier() {}

	@Override
	public void setTrainingData(List<DataEntry> data)
			throws InvalidArgumentException {}

	@Override
	public double regress(double[] x) {
		for(int i = 0; i < x.length; i++){
			this.input.get(i).input(x[i]);
		}
		//Propagate stuff
		for(Neuron n : this.input) n.propagate();
		for(List<Neuron> list : this.hidden){
			for(Neuron n : list) n.propagate();
		}
		for(Neuron n : this.output) n.propagate();
		double[] r = new double[this.output.size()];
		for(int i = 0; i < r.length; i++){
			r[i] = this.output.get(i).getLatestOutput();
		}
		return r[0];
	}
	
	public static AcceptanceCriteria andAcceptanceCriteria(final AcceptanceCriteria ... criterions){
		return new AcceptanceCriteria(){
			@Override
			public boolean isAccepted(int epoch, double mserror, double trainMsError) {
				boolean accept = true;
				for(AcceptanceCriteria a : criterions){
					accept &= a.isAccepted(epoch, mserror, trainMsError);
				}
				return accept;
			}
		};
	}
	
	public static AcceptanceCriteria orAcceptanceCriteria(final AcceptanceCriteria ... criterions){
		return new AcceptanceCriteria(){
			@Override
			public boolean isAccepted(int epoch, double mserror, double trainMsError) {
				boolean accept = true;
				for(AcceptanceCriteria a : criterions){
					if(a.isAccepted(epoch, mserror, trainMsError)){
						System.out.println(a.isAccepted(epoch, mserror, trainMsError));
						return true;
					}
				}
				return accept;
			}
		};
	}
	
	public static AcceptanceCriteria maxEpochAcceptanceCriteria(final int limit){
		return new AcceptanceCriteria(){
			@Override
			public boolean isAccepted(int epoch, double mserror, double trainMsError) {
				boolean retval =epoch >= limit; 
				return retval;
			}
		};
	}
	
	public static AcceptanceCriteria maxRmsAcceptanceCriteria(final double limit){
		return new AcceptanceCriteria(){
			@Override
			public boolean isAccepted(int epoch, double mserror, double trainMsError) {
				return mserror < limit;
			}
		};
	}
	
	public static AcceptanceCriteria plotterAcceptanceCriteria(final String outputFile, final int epochs) throws FileNotFoundException, UnsupportedEncodingException{
		return new AcceptanceCriteria(){
			PrintWriter writer = new PrintWriter(outputFile, "UTF-8");
			@Override
			public boolean isAccepted(int epoch, double mserror, double trainMsError) {
				if(epoch >= epochs){
					writer.close();
					return true;
				}
				String print = epoch+"\t"+trainMsError+"\t"+mserror;
				writer.println(print.replace(".", ","));
				//And add old
				return false;
			}
		};
	}
	
	public static AcceptanceCriteria rmsVariationAcceptanceCriteria(final double limit){
		final LinkedList<Double> errors = new LinkedList<Double>();
		return new AcceptanceCriteria(){
			private int errorSize = 30;
			
			@Override
			public boolean isAccepted(int epoch, double mserror, double trainMsError) {
				errors.addLast(mserror);
				if(errors.size() == this.errorSize+1) errors.removeFirst();
				if(Statistics.variance(errors) < limit && errors.size() == this.errorSize) return true;
				//And add old
				return false;
			}
		};
	}
	
	public static ClassificationAcceptanceCriteria printer(final int limit){
		return new ClassificationAcceptanceCriteria(){
			@Override
			public boolean isAccepted(int epoch, double error, double trainError) {
				if(epoch%100 == 0) System.out.println(epoch+" of: "+limit+", error: "+error+", train error: "+trainError);
				if(epoch > limit) return true;
				return false;
			}
		};
		
	}

	@Override
	public Map<Object, Double> calculateProbabilityForClassifications(double[] x) {
		// TODO Auto-generated method stub
		return null;
	}
}
