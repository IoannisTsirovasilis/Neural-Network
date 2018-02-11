package neuralNetwork;

public class MathTool {

	
	// activation function or sigmoid function
		public static double sigmoid(double z) {
			return 1/(1+Math.exp(-z));
		}	
		
		
	/* Below there are some methods that are currently under development. Java
	 * does not include built-in libraries that allow matrix operations.
	 * Matrixes and vectors simplify the neural network to a great extent.
	 * So, one could ignore the code below this comment or get ideas on how
	 * to improve their network.
	 */
		
	// inner product or dot product
	public static double ip(double[] x, double[] y) {
		if (x.length != y.length) {
			System.out.println("The vectors have different dimensions. -1 returned.");
			return -1;
		}
		double ip = 0;
		for (int i = 0; i < x.length; i++) {
			ip += x[i]*y[i];
		}
		return ip;
	}
	
	// Hadamard product arrays
	public static double[][] hp(double[][] x, double[][] y) {
		if ((x.length != y.length) || (x[0].length != y[0].length)) {
			System.out.println("The dimensions do not match. Null returned.");
			return null;
		}
		double[][] hp = new double[x.length][x[0].length];
		for (int row = 0; row < x.length; row++) {
			for (int col = 0; col < x[0].length; col++) {
				hp[row][col] = x[row][col]*y[row][col];
			}
		}
		return hp;
	}
	
	// Hadamard product vectors
	public static double[] hp(double[] x, double[] y) {
		if (x.length != y.length) {
			System.out.println("The vectors have different dimensions. Null returned.");
			return null;
		}
		double[] hp = new double[x.length];
		for (int row = 0; row < x.length; row++) {
			hp[row] = x[row]*y[row];
		}
		return hp;
	}
	// Transpose of matrix
	public static double[][] transpose(double[][] x) {
		double[][] transpose = new double[x[0].length][x.length];
		for (int row = 0; row < transpose.length; row++) {
			for (int col = 0; col < transpose[0].length; col++) {
				transpose[row][col] = x[col][row];
			}
		}
		return transpose;
	}
	
	// Matrix product
	public static double[][] mp(double[][] x, double[][] y) {
		if (x[0].length != y.length) {
			System.out.println("The dimensions are invalid. Null returned.");
			return null;
		}
		double[][] mp = new double[x.length][y[0].length];
		for (int row = 0; row < mp.length; row++) {
			for (int col = 0; col < mp[0].length; col++) {
				for (int k = 0; k < x[0].length; k++) {
					mp[row][col] += x[row][k]*y[k][col]; 
				}				
			}
		}
		return mp;
	}	
}