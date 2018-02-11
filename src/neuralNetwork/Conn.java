package neuralNetwork;

public class Conn {
	protected double weight, deltaWeight, sDeltaWeight;
	protected Neuron start, end;
	
	public Conn(double weight, Neuron start, Neuron end) {
		this.weight = weight;
		this.start = start;
		this.end = end;
		this.deltaWeight = 0;
	}
	
	protected void updateWeight(double lr, int m) {
		weight -= lr*sDeltaWeight/m;
		sDeltaWeight = 0;
	}
}