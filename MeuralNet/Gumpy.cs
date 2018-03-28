using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MeuralNet
{
    class Gumpy
    {
        public static double[,] Randn(int i, int j)
        {
            double[,] output = new double[i, j];

            return output;
        }

        public static double[] Randn(int i)
        {
            double[] output = new double[i];

            return output;
        }

        public static double[,] Transpose(double[,] input)
        {
            double[,] output = new double[input.GetLength(1), input.GetLength(0)];
            for (int y = 0; y < input.GetLength(0); y++)
            {
                for (int x = 0; x < input.GetLength(1); x++)
                {
                    output[x, y] = input[y, x];
                }
            }
            return output;
        }

        public static double[,] Dot(double[,] left, double[,] right)
        {
            if (left.GetLength(1) != right.GetLength(0)) return null;

            double[,] output = new double[left.GetLength(0), right.GetLength(1)];
            for (int y = 0; y < output.GetLength(0); y++)
            {
                for (int x = 0; x < output.GetLength(1); x++)
                {
                    double total = 0;
                    for (int i = 0; i < left.GetLength(1); i++)
                    {
                        total += (left[y, i] * right[i, x]);
                    }
                    output[y, x] = total;
                }
            }
            return output;
        }

        public static double[,] Multiply(double[,] left, double[,] right)
        {
            if ((left.GetLength(0) != right.GetLength(0)) || (left.GetLength(1) != right.GetLength(1))) return null;

            double[,] output = new double[left.GetLength(0), left.GetLength(1)];
            for (int y = 0; y < output.GetLength(0); y++)
            {
                for (int x = 0; x < output.GetLength(1); x++)
                {
                    output[y, x] = left[y, x] * right[y, x];
                }
            }
            return output;
        }

        public static double[,] Multiply(double[,] left, double coefficient)
        {
            double[,] output = new double[left.GetLength(0), left.GetLength(1)];
            for (int y = 0; y < output.GetLength(0); y++)
            {
                for (int x = 0; x < output.GetLength(1); x++)
                {
                    output[y, x] = left[y, x] * coefficient;
                }
            }
            return output;
        }

        public static double[,] Add(double[,] left, double[,] right)
        {
            if ((left.GetLength(0) != right.GetLength(0)) || (left.GetLength(1) != right.GetLength(1))) return null;

            double[,] output = new double[left.GetLength(0), left.GetLength(1)];
            for (int y = 0; y < output.GetLength(0); y++)
            {
                for (int x = 0; x < output.GetLength(1); x++)
                {
                    output[y, x] = left[y, x] + right[y, x];
                }
            }
            return output;
        }

        public static double[,] Subtract(double[,] left, double[,] right)
        {
            if ((left.GetLength(0) != right.GetLength(0)) || (left.GetLength(1) != right.GetLength(1))) return null;

            double[,] output = new double[left.GetLength(0), left.GetLength(1)];
            for (int y = 0; y < output.GetLength(0); y++)
            {
                for (int x = 0; x < output.GetLength(1); x++)
                {
                    output[y, x] = left[y, x] - right[y, x];
                }
            }
            return output;
        }

        public static void PrintMatrix(double[,] input)
        {
            for (int y = 0; y < input.GetLength(0); y++)
            {
                for (int x = 0; x < input.GetLength(1) - 1; x++)
                {
                    System.Diagnostics.Debug.Write(input[y, x] + ", ");
                }
                System.Diagnostics.Debug.WriteLine(input[y, input.GetLength(1) - 1]);
            }
            System.Diagnostics.Debug.WriteLine("");
        }
    }
}
