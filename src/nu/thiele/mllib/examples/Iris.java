package nu.thiele.mllib.examples;

import nu.thiele.mllib.classifiers.IClassifier;
import nu.thiele.mllib.classifiers.LinearDiscriminantAnalysis;
import nu.thiele.mllib.classifiers.MultiLayerPerceptron;
import nu.thiele.mllib.classifiers.NaiveBayes;
import nu.thiele.mllib.classifiers.NearestNeighbour;
import nu.thiele.mllib.data.Data;
import nu.thiele.mllib.data.DataSet;
import nu.thiele.mllib.utils.Testing;
import nu.thiele.mllib.utils.Testing.ClassifierResults;

/**
 * Iris data set from https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
 * Bache, K. & Lichman, M. (2013). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.
 * 
 * @author Andreas Thiele
 *
 */
public class Iris {
	public static void main(String[] args) throws Exception{
	    DataSet dataset = Data.getIrisTrainingSet();
	    DataSet testset = Data.getIrisTestSet();
	    /*
	     * Usage of single classifier
	     */
	    LinearDiscriminantAnalysis lda = new LinearDiscriminantAnalysis();
	    NearestNeighbour nn = new NearestNeighbour(2);
	    NaiveBayes nb = new NaiveBayes();
	    MultiLayerPerceptron mlp = new MultiLayerPerceptron(1, 0.5);
	    mlp.build(4, new int[]{10}, 3);

	    IClassifier[] classifiers = {lda, nn, nb, mlp};
	    for(IClassifier classifier : classifiers){
	    	classifier.train(dataset.x, dataset.y);
	    }
	    /*
	     * Using testing tools
	     */
	    System.out.println();
	    System.out.println("Using test set");
	    for(IClassifier cl : classifiers){
	    	ClassifierResults ldaR = Testing.testSet(cl, testset);
	    	System.out.println(cl.getClass()+" results: "+ldaR);
	    	ldaR = Testing.crossValidation(cl, dataset, 10);
	    	System.out.println(cl.getClass()+" cross validation: "+ldaR);
	    }
	}
}
