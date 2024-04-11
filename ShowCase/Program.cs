
using SamSharp;
using OpenCvSharp;
using System.Windows.Forms.Design;

class Program {


    [STAThread]
    static void Main(string[] args)
    {
        System.Windows.Forms.OpenFileDialog dialog = new();
        dialog.Title = "Select Encoder Model File";
       

        dialog.Filter = "onnx files (*.onnx)|*.onnx";
        dialog.ShowDialog();

        string encoderPath = dialog.FileName;

        dialog.Title = "Select Decoder Model File";
        dialog.ShowDialog();

        string decoderPath = dialog.FileName;

        dialog.Title = "Select Test Image";
        dialog.Filter = "JPEG Files (*.jpg)|*.jpg";
        dialog.ShowDialog();

        string imagePath = dialog.FileName;


        Console.WriteLine("Initializing Model...");

        using SamInferenceSession sam = new SamInferenceSession(encoderPath, decoderPath);
        sam.Initialize();

        string windowName = "Segment Anything (Click different parts of image to segment)";
        Console.WriteLine("Setting image...");
        sam.SetImage(imagePath);

        var displayImage = new Mat(imagePath);
        var baseImage = new Mat(imagePath);
        var window = new Window(windowName);
        Cv2.ImShow(windowName, displayImage);
       
        Cv2.SetMouseCallback(windowName, (mouseEvent, xCoord, yCoord, flags, ptr) =>
        {
            if (mouseEvent == MouseEventTypes.LButtonDown)
            {
                var mask = sam.GetPointMask(xCoord, yCoord);

                displayImage.Dispose();
                displayImage = new Mat(new OpenCvSharp.Size(baseImage.Width, baseImage.Height), baseImage.Type());

                int pixel = 0;
                for (int y = 0; y < baseImage.Rows; y++)
                {
                    for (int x = 0; x < baseImage.Cols; x++)
                    {
                        double vibrance = mask[pixel++] > 0 ? 1.0 : 0.5;
                        displayImage.At<Vec3b>(y, x) = baseImage.At<Vec3b>(y, x) * vibrance;

                    }
                }

                Cv2.ImShow(windowName, displayImage);
            }
        });

        int tmp = Cv2.WaitKey();

    }

}