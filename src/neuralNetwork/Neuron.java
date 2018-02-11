package neuralNetwork;

public class Neuron {
	protected double bias, activation, deltaBias, sDeltaBias;
	
	public Neuron(double bias) {
		this.bias = bias;
		this.activation = 0;
		this.deltaBias = 0;
	}
	
	protected void updateBias(double lr, int m) {
		bias -= lr*sDeltaBias/m;
		sDeltaBias = 0;
	}
}