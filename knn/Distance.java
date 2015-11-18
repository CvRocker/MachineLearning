package knn;

/**
 * To store the distance of  train data label with given test data.
 * @author leo
 *
 */
public class Distance {

	public double distance;
	
	public int label;
	
	public Distance(int label, double distance){
		this.label = label;
		this.distance = distance;
	}
}
