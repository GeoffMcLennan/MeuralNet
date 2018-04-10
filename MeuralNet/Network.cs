using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml;

namespace MeuralNet
{
    class Network
    {
        private int NumLayers;
        private int[] Sizes;
        private double[][,] Biases;
        private double[][,] Weights;

        private GaussianRandom rand;

        public Network(int[] sizes)
        {
            NumLayers = sizes.Length;
            Sizes = sizes;

            NumLayers = sizes.Length;
            Sizes = sizes;

            // Creates number generator from a gaussian distribution centered at 0 with
            // std dev of 1
            rand = new GaussianRandom(0, 1);

            // Generates lists of biases for all layers except for the inputs
            Biases = new double[NumLayers - 1][,];
            for (int i = 1; i < NumLayers; i++)
            {
                Biases[i - 1] = rand.Sample(1, Sizes[i]);
            }

            // Generates lists of weights relating each layer to the next layer
            Weights = new double[NumLayers - 1][,];
            for (int i = 1; i < NumLayers; i++)
            {
                Weights[i - 1] = rand.Sample(Sizes[i - 1], Sizes[i]);
            }
        }

        public Network(String filename)
        {
            XmlDocument xml = new XmlDocument();
            xml.Load(filename);
            XmlNodeList neuralNet = xml.SelectNodes("NeuralNet");
            XmlNodeList layers = neuralNet[0].SelectNodes("Layer");
            NumLayers = layers.Count + 1;
            Sizes = new int[NumLayers];
            Biases = new double[layers.Count][,];
            Weights = new double[layers.Count][,];

            for (int i = 0; i < layers.Count; i++)
            {
                XmlNodeList weights = layers[i].SelectNodes("Weights");
                XmlNodeList rows = weights[0].SelectNodes("Row");
                Sizes[i] = rows[i].SelectNodes("Col").Count;
                Sizes[i + 1] = rows.Count;
                Weights[i] = new double[Sizes[i + 1], Sizes[i]];
                Biases[i] = new double[Sizes[i + 1], 1];

                XmlNodeList biases = layers[i].SelectNodes("Biases");
                XmlNodeList bRows = biases[0].SelectNodes("Row");

                for (int y = 0; y < Sizes[i + 1]; y++)
                {
                    XmlNodeList cols = rows[y].SelectNodes("Col");
                    for (int x = 0; x < Sizes[i]; x++)
                    {
                        Weights[i][y, x] = Double.Parse(cols[x].InnerText);
                    }

                    Biases[i][y, 0] = Double.Parse(bRows[y].InnerText);
                }
            }
        }

        public void SGD(List<Tuple<double[,], int>> trainingData, int epochs, int batchSize, double learningRate, List<Tuple<double[,], int>> testData)
        {
            for (int i = 0; i < epochs; i++)
            {
                Shuffle(trainingData);
                for (int j = 0; j < trainingData.Count / batchSize; j++)
                {
                    List<Tuple<double[,], int>> batch;
                    if (j * batchSize + batchSize >= trainingData.Count)
                    {
                        batch = trainingData.GetRange(j * batchSize, trainingData.Count - j * batchSize);
                    }
                    else
                    {
                        batch = trainingData.GetRange(j * batchSize, batchSize);
                    }
                    UpdateBatch(batch, learningRate);
                }

                if (testData != null)
                {
                    System.Diagnostics.Debug.WriteLine("Epoch " + i + ": " + Evaluate(testData) + " / " + testData.Count);
                }
                else
                {
                    System.Diagnostics.Debug.WriteLine("Epoch " + i + " complete");
                }
            }
        }

        public void UpdateBatch(List<Tuple<double[,], int>> batch, double learningRate)
        {
            double[][,] nablaB = GenBlankBias();
            double[][,] nablaW = GenBlankWeight();

            foreach (Tuple<double[,], int> t in batch)
            {
                BackProp(nablaB, nablaW, t.Item1, t.Item2);
            }

            double factor = learningRate / batch.Count;
            for (int i = 0; i < Weights.Length; i++)
            {
                Biases[i] = Gumpy.Subtract(Biases[i], Gumpy.Multiply(nablaB[i], factor));
                Weights[i] = Gumpy.Subtract(Weights[i], Gumpy.Multiply(nablaW[i], factor));
            }
        }

        /*
         * Propagate forward and calculate the difference from the expected value.
         * Propagate backwards and add the nabla values to the sum.
         */
         public void BackProp(double[][,] nablaBSum, double[][,] nablaWSum, double[,] input, int value)
        {
            // Blank matrices to be added to sums
            double[][,] nablaB = GenBlankBias();
            double[][,] nablaW = GenBlankWeight();

            // Current activation is the raw input, add it into activations array for safe-keeping
            double[,] activation = input;
            double[][,] activations = new double[Sizes.Length][,];
            activations[0] = activation;
            double[][,] zs = new double[Sizes.Length - 1][,];

            for (int i = 0; i < Sizes.Length - 1; i++)
            {
                double[,] z = Gumpy.Add(Gumpy.Dot(Weights[i], activation), Biases[i]);
                // Add raw layer values to z's array;
                zs[i] = z;
                // Calculate Sigmoid'd values and add to activations array
                activation = Sigmoid(z);
                activations[i + 1] = activation;
            }

            // Prime values for the output layer
            double[,] prime = SigmoidPrime(zs[zs.Length - 1]);
            // Calculated output layer values
            double[,] output = activations[activations.Length - 1];
            double[,] expected = new double[output.Length, 1];
            // index corresponding to expected value should be 1.0, others 0.0
            expected[value, 0] = 1.0;
            // Get the difference between the output and expected data
            double[,] delta = Gumpy.Multiply(Gumpy.Subtract(output, expected), prime);

            // Start the reverse propagation
            nablaB[nablaB.Length - 1] = delta;
            nablaW[nablaW.Length - 1] = Gumpy.Dot(delta, Gumpy.Transpose(activations[activations.Length - 2]));

            for (int i = 2; i < NumLayers; i++)
            {
                prime = SigmoidPrime(zs[zs.Length - i]);
                double[,] wT = Gumpy.Transpose(Weights[Weights.Length - i + 1]);
                delta = Gumpy.Multiply(Gumpy.Dot(wT, delta), prime);

                nablaB[nablaB.Length - i] = delta;
                nablaW[nablaW.Length - i] = Gumpy.Dot(delta, Gumpy.Transpose(activations[activations.Length - i - 1]));
            }

            // Add nablas to sums
            for (int i = 0; i < nablaB.Length; i++)
            {
                nablaBSum[i] = Gumpy.Add(nablaBSum[i], nablaB[i]);
                nablaWSum[i] = Gumpy.Add(nablaWSum[i], nablaW[i]);
            }
        }

        /*
         * Calculates output layer from the given input layer using the 
         * current weights and biases
         */
        public double[,] FeedForward(double[,] rawInput)
        {
            double[,] output = rawInput;
            for (int i = 0; i < NumLayers - 1; i++)
            {
                output = Sigmoid(Gumpy.Add(Gumpy.Dot(Weights[i], output), Biases[i]));
            }
            return output;
        }

        public int Evaluate(double[,] rawInput)
        {
            double[,] output = FeedForward(rawInput);
            int maxIndex = 0;
            double maxVal = output[0, 0];
            for (int j = 1; j < output.GetLength(0); j++)
            {
                if (output[j, 0] > maxVal)
                {
                    maxIndex = j;
                    maxVal = output[j, 0];
                }
            }
            return maxIndex;
        }

        public int Evaluate(List<Tuple<double[,], int>> input)
        {
            int count = 0;
            for (int i = 0; i < input.Count; i++)
            {
                double[,] output = FeedForward(input[i].Item1);
                int maxIndex = 0;
                double maxVal = output[0, 0];
                for (int j = 1; j < output.GetLength(0); j++)
                {
                    if (output[j, 0] > maxVal)
                    {
                        maxIndex = j;
                        maxVal = output[j, 0];
                    }
                }
                if (i % 50 == 0)
                {
                    Console.WriteLine("Test " + i + ": Guessed " + maxIndex + ", Actual " + input[i].Item2);
                }
                if (maxIndex == input[i].Item2) count++;
            }
            return count;
        }

        public void Save(String filename)
        {
            XmlDocument output = new XmlDocument();
            XmlNode root = output.CreateElement("NeuralNet");
            output.AppendChild(root);

            for (int i = 0; i < Sizes.Length - 1; i++)
            {
                XmlNode layer = output.CreateElement("Layer");
                XmlNode weight = output.CreateElement("Weights");
                XmlNode bias = output.CreateElement("Biases");
                layer.AppendChild(weight);
                layer.AppendChild(bias);

                for (int y = 0; y < Weights[i].GetLength(0); y++)
                {
                    XmlNode wRow = output.CreateElement("Row");

                    for (int x = 0; x < Weights[i].GetLength(1); x++)
                    {
                        // Add weight
                        XmlNode col = output.CreateElement("Col");
                        col.InnerText = "" + Weights[i][y, x];
                        wRow.AppendChild(col);
                    }
                    weight.AppendChild(wRow);
                }

                for (int j = 0; j < Biases[i].GetLength(0); j++)
                {
                    XmlNode b = output.CreateElement("Row");
                    b.InnerText = "" + Biases[i][j, 0];
                    bias.AppendChild(b);
                }
                root.AppendChild(layer);
            }

            output.Save(filename);
        }

        public static double Sigmoid(double z)
        {
            return 1.0 / (1.0 + Math.Exp(-z));
        }

        public static double[,] Sigmoid(double[,] input)
        {
            double[,] output = new double[input.GetLength(0), input.GetLength(1)];
            for (int y = 0; y < output.GetLength(0); y++)
            {
                for (int x = 0; x < output.GetLength(1); x++)
                {
                    output[y, x] = Sigmoid(input[y, x]);
                }
            }
            return output;
        }

        public static double SigmoidPrime(double z)
        {
            return Sigmoid(z) * (1 - Sigmoid(z));
        }

        public static double[,] SigmoidPrime(double[,] input)
        {
            double[,] output = new double[input.GetLength(0), input.GetLength(1)];
            for (int i = 0; i < output.Length; i++)
            {
                for (int j = 0; j < output.GetLength(1); j++)
                {
                    output[i, j] = SigmoidPrime(input[i, j]);
                }
            }
            return output;
        }

        public void Shuffle<T>(List<T> input)
        {
            Random rnd = new Random();
            input = input.OrderBy(x => rnd.Next()).ToList();
        }

        public double[][,] GenBlankBias()
        {
            double[][,] output = new double[Sizes.Length - 1][,];
            for (int i = 0; i < Sizes.Length - 1; i++)
            {
                output[i] = new double[Sizes[i + 1], 1];
            }
            return output;
        }

        public double[][,] GenBlankWeight()
        {
            double[][,] output = new double[Sizes.Length - 1][,];
            for (int i = 0; i < Sizes.Length - 1; i++)
            {
                output[i] = new double[Sizes[i + 1], Sizes[i]];
            }
            return output;
        }
    }

    class NetworkTester
    {
        public static void Test()
        {
            DataParser training = new DataParser("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
            DataParser testing = new DataParser("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");

            Network net = new Network(new int[] { 784, 30, 10 });
            //Network net = new Network("BrainChild4.xml");
            net.SGD(training.GetData()/*.GetRange(0, 10000)*/, 1, 10, 3.0, testing.GetData());

            //System.Diagnostics.Debug.WriteLine(net.Evaluate(testing.GetData()) + " / " + testing.GetData().Count);

            net.Save("BestBrainchild3.xml");

            //Network net = new Network(new int[] { 3, 5, 2 });
            //double[,] input = new double[,] { { 3 }, { 4 }, { 5 } };
            //double[,] test = net.FeedForward(input);
            //Print(test);
            //List<Tuple<double[,], int>> list = new List<Tuple<double[,], int>>();
            //list.Add(Tuple.Create(input, 1));
            //list.Add(Tuple.Create(new double[,] { { 1 }, { 2 }, { 3 } }, 0));
            //net.UpdateBatch(list, 3.0);
        }

        public static void Print(double[,] input)
        {
            for (int y = 0; y < input.GetLength(0); y++)
            {
                for (int x = 0; x < input.GetLength(1) - 1; x++)
                {
                    Console.Write(input[y, x] + ", ");
                }
                Console.WriteLine(input[y, input.GetLength(1) - 1]);
            }
        }
    }
}
