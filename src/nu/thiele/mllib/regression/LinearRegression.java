package nu.thiele.mllib.regression;

import java.util.List;

import nu.thiele.mllib.data.Data.DataEntry;
import Jama.Matrix;
import Jama.QRDecomposition;

public class LinearRegression implements IRegressor{
	private double alpha, beta = 1.0; //Beta = 1
	private double[] w;
	private Matrix designMatrix, mn, sn;
	private Solution solution;
	
	public LinearRegression(List<DataEntry> data, double alpha, double beta){
		this.solution = Solution.A_POSTERIORI;
		this.alpha = alpha;
		this.beta = beta;
		double[] y = new double[data.size()];
		double[][] matrix = new double[data.size()][data.get(0).getX().length];
		int k = 0;
		for(DataEntry d : data){
			y[k] = Double.valueOf(d.getY().toString());
			matrix[k] = d.getX();
			k++;
		}
		
		//First, I need to add a column first for the bias
		//Very ugly and difficult to understand with the indexes
		double[][] nmatrix = new double[matrix.length][matrix[0].length+1];
		for(int i = 0; i < matrix.length; i++){
			nmatrix[i][0] = 1;
			for(int j = 1; j < nmatrix[0].length; j++){
				nmatrix[i][j] = matrix[i][j-1];
			}
		}
		
		double[] t = new double[data.size()];
		int i = 0;
		for(DataEntry d : data){
			t[i] = Double.valueOf(d.getY().toString());
			i++;
		}
		Matrix yMatrix = new Matrix(new double[][]{t}).transpose();

		//And do stuff
		this.designMatrix = new Matrix(matrix);
		Matrix identity = Matrix.identity(this.designMatrix.getColumnDimension(), this.designMatrix.getColumnDimension());
		Matrix snInv = identity.times(this.alpha).plus(this.designMatrix.transpose().times(this.designMatrix).times(beta));
		this.sn = snInv.inverse(); //Inverse() calculates pseudoinverse if not square. Confusing
		this.mn = this.sn.times(this.designMatrix.transpose()).times(yMatrix).times(this.beta).transpose();
	}
	
	/**
	 * Use this for maximum likelihood estimation
	 * @param data
	 */
	public LinearRegression(List<DataEntry> data){
		this.solution = Solution.LIKELIHOOD;
		double[] y = new double[data.size()];
		double[][] matrix = new double[data.size()][data.get(0).getX().length];
		int k = 0;
		for(DataEntry d : data){
			y[k] = Double.valueOf(d.getY().toString());
			matrix[k] = d.getX();
			k++;
		}
		//First, I need to add a column first for the bias
		//Very ugly and difficult to understand with the indexes
		double[][] nmatrix = new double[matrix.length][matrix[0].length+1];
		for(int i = 0; i < matrix.length; i++){
			nmatrix[i][0] = 1;
			for(int j = 1; j < nmatrix[0].length; j++){
				nmatrix[i][j] = matrix[i][j-1];
			}
		}
		
		Matrix mx = new Matrix(nmatrix);
		Matrix my = new Matrix(y, y.length);
		
		QRDecomposition qr = new QRDecomposition(mx);
		double[][] weights = qr.solve(my).getArray();
		this.w = new double[weights.length];
		for(int i = 0; i < this.w.length; i++){
			this.w[i] = weights[i][0];
		}		
	}
	
	public double[] getParameters(){
		return this.w;
	}
	
	@Override
	public double regress(double[] x) {
		switch(this.solution){
		default:
		case LIKELIHOOD:
			double retval = this.w[0]; //Start with bias
			for(int i = 1; i < this.w.length; i++){
				retval += x[i-1]*this.w[i];
			}
			return retval;
		case A_POSTERIORI:
			double[][] v = new double[1][];
			v[0] = x;
			Matrix xVector = new Matrix(v);
			Matrix classification = this.mn.times(xVector.transpose());
			return classification.get(0,0);
		}
	}
	
	public static enum Solution{
		A_POSTERIORI, LIKELIHOOD
	}
}
