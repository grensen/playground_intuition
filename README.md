# Playground Intuition

<p align="center">
  <img src="https://github.com/grensen/playground_intuition/blob/main/examples/playground_intuition_intro.png">
</p>

Wait before you go, this is from Google. Even though it's not an official Google product, you can use the [Playground](https://playground.tensorflow.org/) to understand basic neural networks. If you're new to this field, you'll need some intuition, otherwise what's coming next might be a bit too much, even if it seems simple after you learned it. 

But before it really starts there are two things to consider. X1 and X2 represent the input coordinates on the prediction map. The usual way to describe the points on the map would be to take X1 = X for the data point from left to right, and X2 = Y for the data point from top to bottom. The center alignment with 0 in the middle is also important. Without this center, the shifts would still have to be taken into account.

The bias can be seen when using deep neural networks as shown in the figure. Unfortunately, the output neuron was apparently forgotten and does not offer the possibility to set the bias. That means to understand everything correctly we have to add a bias of 0.1 to every equation, even if the output neuron seems not there. Everything here aims to understand the cores in deep learning, the perceptrons.

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

Not very nice written code, but understandable enough that you can see with a trained eye that what is happening is what you learn in theory about how neural networks work. Neural networks are simply a bunch of perceptrons, and deep neural networks stack several layers of these bunches on top of each other.

The main equation is always input neuron * weight + bias = output neuron. Fortunately, as complicated as everything may seem at first sight, this is completely sufficient to understand everything that comes next.

## Basic Understanding

<p align="center">
  <img src="https://github.com/grensen/playground_intuition/blob/main/examples/basic_understanding.png">
</p>

This could now be seen as a perceptron, input neurons, in this case only one, are connected with weights to an output neuron. Instead of hidden activations, the output neuron is activated directly. This is difficult to see in the Playground code.
But you can find it right [here](https://github.com/tensorflow/playground/blob/master/src/heatmap.ts#L168):
~~~ts
if (discretize) {
  value = (value >= 0 ? 1 : -1);
}
~~~

This code creates the prediction map that calculates the background to see how the network predicts the data points. That means if the output neuron is greater than or equal to zero, the prediction for positive results is blue. If the prediction was less than zero, the point is gold. 

In addition, "Discretize output" is clicked to make the prediction clearly visible, whether it is class 1 = point is blue, or whether it is class -1 = point is gold. The whole thing is then called a binary classification problem. The input feature takes the X coordinates and squared it (X * X = X²).

For example, let's take a point with 1 on the x-axis, first the feature must be created with X² = 1 * 1 = 1. Then the calculation is 1 * -0.1 = -0.1, this would be the output neuron. But we have to add a bias with 0.1 how the playground, -0.1 + 0.1 = 0, yes, dies is the output neuron, which predicts with (value = (value >= 0 ? 1 : -1)), in words, if the output value is 0 or greater than 0, we predict a blue point as class 1.

But what if we take a point with X = -1? We square again first with -1 * -1 = 1, and then simply calculate 1 * -0.1 + 0.1 = 0 (input * weight + bias = output). Which also predicts the data point as class blue. Hopefully, you can clearly see how this creates the prediction map, but also that the prediction can only work if you add the bias term. But to really solve the problem we need a second feature.

## Circle Solution

<p align="center">
  <img src="https://github.com/grensen/playground_intuition/blob/main/examples/circle_solution.png">
</p>

With the second feature and slightly adjusted with smaller weights this prediction map is created which solves the problem. Smaller weights therefore increase the prediction circle for class blue. From the point of view of the solution, a golden point should result in a negative output value. 

For example, data point with X = -4 and Y = 2 is a golden data point. First we create the two features with X² = 16 and Y² = 4.
Then if we calculate (16 * -0.016) + (4 * -0.015) + 0.1 we get -0.216 as output value. And if the output is less than zero, it is classified as class gold. Cool!

## xor

<p align="center">
  <img src="https://github.com/grensen/playground_intuition/blob/main/examples/xor.png">
</p>

Perhaps by this point, some "aha" moments have already happened. But just to be sure let's solve the simplest non-linear example, the XOR problem. Here the created features are again the key. For the blue data point X = -4 and Y = -5 the calculation is (-4 * -5 = 20) * 1 + 0.1 = 20.1 which is a positive output value and therefore class prediction for this data point is blue.

## But why not a clean solution?

<p align="center">
  <img src="https://github.com/grensen/playground_intuition/blob/main/examples/xor_clean.png">
</p>

This solution is already much better, but you have to do it by yourself and increase the weight. The result is then a clean solution to the problem. However, this was only necessary because the bias influenced the prediction. This example also shows where a bias distorts the solution somewhat and thus increases the error. Unlike the circle dataset where a bias is necessary to solve the problem. 

## spiral_failed

<p align="center">
  <img src="https://github.com/grensen/playground_intuition/blob/main/examples/spiral_failed.png">
</p>

Now it becomes harder, spirals were considered unsolvable for a long time, especially without hidden neurons, since we don't have any here. Even though we are far from the solution, this is already an extremely interesting picture. All input features together manage to curve the prediction on the map. At this point I will leave the playground and go one step further. 

## What if...
