using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MeuralNet
{
    class GaussianRandom
    {
        private double Mean;
        private double StdDev;
        private Random rand;

        public GaussianRandom(double mean, double stdDev)
        {
            Mean = mean;
            StdDev = stdDev;

            rand = new Random();
        }

        public double Sample()
        {
            double u1 = 1 - rand.NextDouble();
            double u2 = 1 - rand.NextDouble();
            double randStdNormal = Math.Sqrt(-2 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);

            return Mean + StdDev * randStdNormal;
        }

        public double[] Sample(int length)
        {
            double[] output = new double[length];

            for (int i = 0; i < length; i++)
            {
                output[i] = Sample();
            }

            return output;
        }

        public double[,] Sample(int width, int height)
        {
            double[,] output = new double[height, width];

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    output[y, x] = Sample();
                }
            }

            return output;
        }
    }
}
