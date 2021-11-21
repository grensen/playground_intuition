# Playground Intuition

<p align="center">
  <img src="https://github.com/grensen/playground_intuition/blob/main/examples/playground_intuition_intro.png">
</p>

Wait before you go, this is from Google. Even though it's not an official Google product, you can use the [Playground](https://playground.tensorflow.org/) to understand basic neural networks. If you're new to this field, you'll need some intuition, otherwise what's coming up might be a bit too much, even if it seems simple after you learned it. We are not dealing with the most complicated mathematical constructs here. Most problems can be understood very intuitively and sometimes it requires only a single connection to get the correct solution. 

But before it really starts there are two things to consider. X1 and X2 represent the input coordinates on the prediction map. The usual way to describe the points on the map would be to take X1 = X for the data point from left to right, and X2 = Y for the data point from top to bottom. The center alignment with 0 in the middle is also important. Without this center, the shifts would still have to be taken into account. But here everything remains very theoretical and simple.

The bias can be seen when using deep neural networks as shown in the figure. Unfortunately, the output neuron was apparently forgotten and does not offer the possibility to set the bias. That means to understand everything correctly we have to add a bias of 0.1 to every equation, even if the output neuron seems not there. Everything here aims to understand the cores in deep learning, the perceptrons, which are the basis for what runs around today with such trendy terms.

Tensorflow Playground [code](https://github.com/tensorflow/playground/blob/master/src/nn.ts#L59) example:

~~~ts
  /** Recomputes the node's output and returns it. */
  updateOutput(): number {
    // Stores total input into the node.
    this.totalInput = this.bias;
    for (let j = 0; j < this.inputLinks.length; j++) {
      let link = this.inputLinks[j];
      this.totalInput += link.weight * link.source.output;
    }
    this.output = this.activation.output(this.totalInput);
    return this.output;
  }  
~~~

Not very nice written code, but understandable enough that you can see with a trained eye that what is happening is what you learn in theory about how neural networks work. I would rather describe it in C# like this: 

~~~cs
float ComputeOutputNeuron()
{
    // init network
    float[] inputNeuron = { 1, 1 };
    float[] weight = { -0.016f, -0.015f };
    float bias = 0.1f;

    // calc output
    float output = bias;
    for (int i = 0; i < 1; i++)
        output += inputNeuron[i] * weight[i];

    return output;
}
~~~

The language is not so important, the main thing is to understand the process, which is always the same. 

## basic_understanding

<p align="center">
  <img src="https://github.com/grensen/playground_intuition/blob/main/examples/basic_understanding.png">
</p>

This could now be seen as a perceptron, input neurons, in this case only one, are connected with weights to an output neuron. The output neuron is then usually activated. As with the ReLU function, but we don't do that here.

[ReLU](https://github.com/tensorflow/playground/blob/master/src/nn.ts#L122) from the code:

~~~ts
public static RELU: ActivationFunction = {
  output: x => Math.max(0, x),
  der: x => x <= 0 ? 0 : 1
};
~~~

Instead of hidden activations, the output neuron is activated directly. This is difficult to see in the Playground code.
But you can find it right [here](https://github.com/tensorflow/playground/blob/master/src/heatmap.ts#L168):

~~~ts
let value = data[x][y];
if (discretize) {
  value = (value >= 0 ? 1 : -1);
}
~~~

The good news is, we don't need programm code anymore, a calculator is enough. It was important for me to show that the code, even if it seems ultra complicated, can be understood better than you might have thought at first. 

To the prediction map, let's make the first calculation.....

---

## circle_solution

<p align="center">
  <img src="https://github.com/grensen/playground_intuition/blob/main/examples/circle_solution.png">
</p>

## xor

<p align="center">
  <img src="https://github.com/grensen/playground_intuition/blob/main/examples/xor.png">
</p>

## spiral_failed

<p align="center">
  <img src="https://github.com/grensen/playground_intuition/blob/main/examples/spiral_failed.png">
</p>
