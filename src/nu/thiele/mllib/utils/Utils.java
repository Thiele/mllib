package nu.thiele.mllib.utils;

import java.util.Map;

import Jama.Matrix;

public class Utils {
	public static void printMatrix(Matrix m){
		for(int i = 0; i < m.getRowDimension(); i++){
			for(int j = 0; j < m.getColumnDimension(); j++){
				System.out.print(m.get(i, j)+" ");
			}
			System.out.println();
		}
	}
	
	public static void percentify(Map<Object,Double> map){
		double total = 0;
		double lowest = 0;
		for(Object o : map.keySet()){
			if(map.get(o) < lowest) lowest = map.get(o);
			total += map.get(o);
		}
		if(lowest < 0){ //Something fucked up. Set one to 0
			for(Object o : map.keySet()){
				map.put(o, map.get(o)-lowest);
			}
			total = total-lowest*map.keySet().size();
		}
		for(Object o : map.keySet()){
			map.put(o, map.get(o)/total);
		}
	}
}
