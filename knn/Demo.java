/**
 * @author liuzhi
 *    date 2015.9.16
 */
package knn;

import java.util.Map;

public class Demo {

	static String trainFile = "data/traindata_10000";
	static String testFile = "data/testdata_5000";
	public static void main(String[] args) throws Exception {

		KNN knn = new KNN(trainFile, 10);
		Map<int[], int[]> testData = knn.loadData(testFile);
		knn.classificationAndTestPrecision(testData);
	}

}
