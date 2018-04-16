using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MeuralNet
{
    class DataParser
    {
        List<Tuple<double[,], int>> data;

        /*
         * Parses the MNIST data files
         */
        public DataParser(String images, String labels)
        {
            byte[] imageBytes = File.ReadAllBytes(images);

            byte[] temp = new byte[12];
            temp[0] = imageBytes[4];
            temp[1] = imageBytes[5];
            temp[2] = imageBytes[6];
            temp[3] = imageBytes[7];
            temp[4] = imageBytes[8];
            temp[5] = imageBytes[9];
            temp[6] = imageBytes[10];
            temp[7] = imageBytes[11];
            temp[8] = imageBytes[12];
            temp[9] = imageBytes[13];
            temp[10] = imageBytes[14];
            temp[11] = imageBytes[15];
            Array.Reverse(temp);
            int numImages = BitConverter.ToInt32(temp, 8);
            int numRows = BitConverter.ToInt32(temp, 4);
            int numCols = BitConverter.ToInt32(temp, 0);
            int pixelsPerImg = numRows * numCols;

            byte[] labelBytes = File.ReadAllBytes(labels);
            temp = new byte[4];
            temp[0] = labelBytes[4];
            temp[1] = labelBytes[5];
            temp[2] = labelBytes[6];
            temp[3] = labelBytes[7];
            Array.Reverse(temp);
            int numLabels = BitConverter.ToInt32(temp, 0);
            if (numImages != numLabels) throw new Exception();

            data = new List<Tuple<double[,], int>>();
            for (int i = 0; i < numImages; i++)
            {
                double[,] img = new double[pixelsPerImg, 1];
                for (int j = 0; j < pixelsPerImg; j++)
                {
                    img[j, 0] = imageBytes[i * pixelsPerImg + j + 16] / 255.0;
                }
                int label = labelBytes[i + 8];
                data.Add(Tuple.Create(img, label));
            }
        }

        public List<Tuple<double[,], int>> GetData()
        {
            return data;
        }
    }
}
