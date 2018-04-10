using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace MeuralNet
{
    public partial class Form1 : Form
    {
        Network net;
        private bool drawing;
        List<List<Point>> points;
        DataParser training;
        DataParser testing;

        int totalGuesses;
        int totalCorrect;

        bool hasOutput;

        const int PEN_WIDTH = 20;
        const int PEN_COLOR = 50;
        Color PenColor;

        public Form1()
        {
            InitializeComponent();
            drawing = false;
            points = new List<List<Point>>();
            PenColor = Color.FromArgb(PEN_COLOR, PEN_COLOR, PEN_COLOR);
            totalGuesses = 0;
            totalCorrect = 0;
            hasOutput = false;

            if (System.IO.File.Exists("train-images.idx3-ubyte") && System.IO.File.Exists("train-labels.idx1-ubyte")
                    && System.IO.File.Exists("t10k-images.idx3-ubyte") && System.IO.File.Exists("t10k-labels.idx1-ubyte"))
            {
                training = new DataParser("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
                testing = new DataParser("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");
            }

            if (System.IO.File.Exists("BrainChild.xml"))
            {
                net = new Network("BrainChild.xml");
            } else
            {
                net = new Network(new int[] { 784, 30, 10 });
                if (training != null && testing != null)
                {
                    net.SGD(training.GetData()/*.GetRange(0, 10000)*/, 1, 10, 3.0, testing.GetData());
                    net.Save("BrainChild.xml");
                }
            }

            panel1.MouseDown += new MouseEventHandler(P1MouseDown);
            panel1.MouseUp += new MouseEventHandler(P1MouseUp);
            panel1.MouseMove += new MouseEventHandler(P1Move);

            button1.Click += new EventHandler(EvaluateInput);
            button2.Click += new EventHandler(CorrectClick);
            button3.Click += new EventHandler(IncorrectClick);
            button4.Click += new EventHandler(IgnoreClick);
        }

        public void P1MouseDown(Object sender, MouseEventArgs e)
        {
            drawing = true;
            points.Add(new List<Point>());
            points.Last().Add(new Point(e.X, e.Y));
        }

        public void P1MouseUp(Object sender, MouseEventArgs e)
        {
            drawing = false;
            points.Last().Add(new Point(e.X, e.Y));
        }

        public void P1Move(Object sender, MouseEventArgs e)
        {
            if (drawing)
            {
                points.Last().Add(new Point(e.X, e.Y));

                Graphics g = ((Panel)sender).CreateGraphics();
                g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.HighQuality;
                g.CompositingQuality = System.Drawing.Drawing2D.CompositingQuality.HighQuality;
                g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
                Pen p = new Pen(PenColor, PEN_WIDTH);
                p.StartCap = System.Drawing.Drawing2D.LineCap.Round;
                p.EndCap = System.Drawing.Drawing2D.LineCap.Round;

                for (int i = 0; i < points.Count; i++)
                {
                    g.DrawLines(p, points[i].ToArray());
                }

                g.Dispose();
                p.Dispose();
            }
        }

        public void EvaluateInput(Object sender, EventArgs e)
        {
            Bitmap bmp = new Bitmap(panel1.Width, panel1.Height);
            Graphics g = Graphics.FromImage(bmp);
            g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.HighQuality;
            g.CompositingQuality = System.Drawing.Drawing2D.CompositingQuality.HighQuality;
            g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
            Pen p = new Pen(PenColor, PEN_WIDTH);
            p.StartCap = System.Drawing.Drawing2D.LineCap.Round;
            p.EndCap = System.Drawing.Drawing2D.LineCap.Round;

            for (int i = 0; i < points.Count; i++)
            {
                g.DrawLines(p, points[i].ToArray());
            }

            g.Dispose();
            p.Dispose();

            Bitmap scaled = new Bitmap(bmp, 28, 28);

            UpdateInputPanel(scaled);

            int maxY = 0;
            int minY = 27;
            int maxX = 0;
            int minX = 27;
            for (int y = 0; y < 28; y++)
            {
                for (int x = 0; x < 28; x++)
                {
                    if (scaled.GetPixel(x, y).A > 128)
                    {
                        if (y > maxY) maxY = y;
                        if (y < minY) minY = y;
                        if (x > maxX) maxX = x;
                        if (x < minX) minX = x;
                    }
                }
            }

            int height = maxY - minY;
            int width = maxX - minX;
            int yOffset = (27 - height) / 2;
            int xOffset = (27 - width) / 2;
            int yDiff = yOffset - minY;
            int xDiff = xOffset - minX;

            double[,] input = new double[784, 1];
            for (int y = 0; y < 28; y++)
            {
                for (int x = 0; x < 28; x++)
                {
                    int b = 0;
                    if (x - xDiff >= 0 && x - xDiff < 28 && y - yDiff >= 0 && y - yDiff < 28)
                    {
                        b = scaled.GetPixel(x - xDiff, y - yDiff).A;
                    }
                    input[y * 28 + x, 0] = b / 255.0;
                }
            }

            UpdateInputPanel(input);

            int guess = net.Evaluate(input);
            Console.WriteLine(guess);

            label2.Text = "" + guess;
            hasOutput = true;
        }

        public void UpdateInputPanel(double[,] input)
        {
            Bitmap bmp = new Bitmap(28, 28);
            for (int y = 0; y < 28; y++)
            {
                for (int x = 0; x < 28; x++)
                {
                    int val = 255 - (int)Math.Floor(input[y * 28 + x, 0] * 255);
                    bmp.SetPixel(x, y, Color.FromArgb(val, val, val));
                }
            }
            Graphics g = panel3.CreateGraphics();
            g.Clear(SystemColors.ControlLightLight);
            g.DrawImage(new Bitmap(bmp, panel3.Width, panel3.Height), Point.Empty);
            g.Dispose();
        }
        
        public void UpdateInputPanel(Bitmap input)
        {
            Graphics g = panel3.CreateGraphics();
            g.Clear(SystemColors.ControlLightLight);
            g.DrawImage(new Bitmap(input, panel3.Width, panel3.Height), Point.Empty);
            g.Dispose();
        }

        public void ClearDrawPanel()
        {
            ClearPanel(panel1);

            points = new List<List<Point>>();
        }

        public void ClearPanel(Panel p)
        {
            Graphics g = p.CreateGraphics();
            g.Clear(SystemColors.ControlLightLight);
            g.Dispose();
        }

        public void CorrectClick(Object sender, EventArgs e)
        {
            Console.WriteLine("Correct");
            if (hasOutput)
            {
                totalGuesses++;
                totalCorrect++;
                label1.Text = totalCorrect + " out of " + totalGuesses + " guesses correct";
                ClearDrawPanel();
                ClearPanel(panel3);
                label2.Text = "";
                hasOutput = false;
            }
        }

        public void IncorrectClick(Object sender, EventArgs e)
        {
            Console.WriteLine("Incorrect");
            if (hasOutput)
            {
                totalGuesses++;
                label1.Text = totalCorrect + " out of " + totalGuesses + " guesses correct";
                ClearDrawPanel();
                ClearPanel(panel3);
                label2.Text = "";
                hasOutput = false;
            }
        }

        public void IgnoreClick(Object sender, EventArgs e)
        {
            Console.WriteLine("Ignore");
            if (hasOutput)
            {
                ClearDrawPanel();
                ClearPanel(panel3);
                label2.Text = "";
                hasOutput = false;
            }
        }
    }
}
