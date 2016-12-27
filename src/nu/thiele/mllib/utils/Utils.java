package nu.thiele.mllib.utils;

import java.util.List;
import java.util.Map;

import Jama.Matrix;
import nu.thiele.mllib.data.Data.DataEntry;
import nu.thiele.mllib.data.DataSet;

public class Utils {
	public static void printMatrix(Matrix m){
		for(int i = 0; i < m.getRowDimension(); i++){
			for(int j = 0; j < m.getColumnDimension(); j++){
				System.out.print(m.get(i, j)+" ");
			}
			System.out.println();
		}
	}
	
	public static void percentify(Map<Double,Double> map){
		double total = 0;
		double lowest = 0;
		for(Object o : map.keySet()){
			if(map.get(o) < lowest) lowest = map.get(o);
			total += map.get(o);
		}
		if(lowest < 0){ //Something fucked up. Set one to 0
			for(Double o : map.keySet()){
				map.put(o, map.get(o)-lowest);
			}
			total = total-lowest*map.keySet().size();
		}
		for(Double o : map.keySet()){
			map.put(o, map.get(o)/total);
		}
	}
	
	public static DataSet listToDataSet(List<DataEntry> list){
		DataSet retval = new DataSet();
		int i = 0;
		retval.x = new double[list.size()][];
		retval.y = new double[list.size()];
		for(DataEntry d : list){
			retval.x[i] = d.getX();
			retval.y[i] = d.getY();
			i++;
		}
		return retval;
	}
}
