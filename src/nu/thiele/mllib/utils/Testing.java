package nu.thiele.mllib.utils;

import java.util.LinkedList;
import java.util.List;

import nu.thiele.mllib.classifiers.IClassifier;
import nu.thiele.mllib.data.Data.DataEntry;
import nu.thiele.mllib.exceptions.InvalidArgumentException;
import nu.thiele.mllib.regression.IRegressor;


public class Testing {
	public static ClassifierResults crossValidation(IClassifier classifier, List<DataEntry> dataset, int folds) throws InvalidArgumentException{
		ClassifierResults retval = new ClassifierResults(0,0);
		int setsize = dataset.size()/folds;
		for(int i = 0; i < folds; i++){
			LinkedList<DataEntry> testset;
			LinkedList<DataEntry> trainingset;
			if(i == 0){ //Test set is first
				testset = new LinkedList<DataEntry>(dataset.subList(0, setsize));
				trainingset = new LinkedList<DataEntry>(dataset.subList(setsize, dataset.size()));
			}
			else if(i == folds-1){ //Test set is last
				testset = new LinkedList<DataEntry>(dataset.subList((i)*setsize, dataset.size()));
				trainingset = new LinkedList<DataEntry>(dataset.subList(0, (i-1)*setsize));
			}
			else{ //Test set somewhere in the middle
				testset = new LinkedList<DataEntry>(dataset.subList(setsize*i, (i+1)*setsize));
				trainingset = new LinkedList<DataEntry>(dataset.subList(0, setsize*i));
				trainingset.addAll(dataset.subList((i+1)*setsize, dataset.size()));
			}
			//Prepare classifier
			classifier.setTrainingData(trainingset);
			classifier.loadClassifier();
			
			//And calculate
			ClassifierResults results = Testing.testSet(classifier, testset);
			retval.addCorrectGuesses(results.getCorrectGuesses());
			retval.addTotalGuesses(results.getTotalGuesses());
		}
		return retval;
	}
	
	public static double[] getRegressionValues(IRegressor regression, List<DataEntry> data){
		double[] retval = new double[data.size()];
		int i = 0;
		for(DataEntry d : data){
			retval[i] = regression.regress(d.getX());
			i++;
		}
		return retval;
	}
	
	public static double rootMeanSquareError(IRegressor regression, List<DataEntry> dataset){
		double retval = 0;
		for(DataEntry d : dataset){
			double y = 0;
			if(d.getY() instanceof Double) y = Double.valueOf(d.getY().toString());
			else if(d.getY() instanceof double[]) y = ((double[])d.getY())[0];
			double error = y - regression.regress(d.getX());
			retval += error*error;
		}
		retval = retval/((double)dataset.size());
		return Math.sqrt(retval);
	}
	
	public static ClassifierResults testSet(IClassifier classifier, List<DataEntry> testset){
		ClassifierResults retval = new ClassifierResults(0,0);
		for(DataEntry entry : testset){
			Object y = entry.getY();
			Object classification = classifier.classify(entry.getX());
			if(classification != null && classification.equals(y)) retval.incrementCorrectGuesses();
			retval.incrementTotalGuesses();
		}
		return retval;
	}
	
	
	public static class ClassifierResults{
		private int correct = 0;
		private int total = 0;
		public ClassifierResults(int c, int t){
			this.correct = c;
			this.total = t;
		}
		
		public void addCorrectGuesses(int c){
			this.correct += c;
		}

		public void addTotalGuesses(int t){
			this.total += t;
		}
		
		public void incrementCorrectGuesses(){
			this.correct++;
		}
		
		public void incrementTotalGuesses(){
			this.total++;
		}
		
		public float getAccuracy(){
			return (this.correct/((float)this.total));
		}
		
		public int getCorrectGuesses(){
			return this.correct;
		}
		
		public int getTotalGuesses(){
			return this.total;
		}
		
		public String toString(){
			return "Correct: "+this.correct+" of: "+this.total+". Accuracy: "+this.getAccuracy();
		}
	}
}
