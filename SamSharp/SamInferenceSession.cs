using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.IO;
using System.Linq;

namespace SamSharp
{
    public class SamInferenceSession : IDisposable
    {
        private InferenceSession encoder;
        private InferenceSession decoder;

        private string encoderPath = "";
        private string decoderPath = "";
        private int sideLength = 1024;
        private int oldWidth = 0;
        private int oldHeight = 0;
        private int newWidth = 0;
        private int newHeight = 0;
        private DenseTensor<float> imageEmbedding;
        private bool initalized = false;

        List<NamedOnnxValue> inputList = new();
        public SamInferenceSession(string encoderPath, string decoderPath, int sideLength = 1024)
        {
            this.encoderPath = encoderPath;
            this.decoderPath = decoderPath;
            this.sideLength = sideLength;
        }


        public void Initialize() 
        {
            if (initalized)
            {
                return;
            }

            if (string.IsNullOrEmpty(encoderPath) || !File.Exists(encoderPath) || !encoderPath.EndsWith(".onnx"))
            {
                throw new FileNotFoundException($"Not a valid onnx file at location '{encoderPath}'");
            }


            if (string.IsNullOrEmpty(decoderPath) || !File.Exists(decoderPath) || !decoderPath.EndsWith(".onnx"))
            {
                throw new FileNotFoundException($"Not a valid onnx file at location '{decoderPath}'");
            }

            encoder = new InferenceSession(encoderPath);
            decoder = new InferenceSession(decoderPath);
            InitList();
            initalized = true;
        }

        public void Dispose()
        {
            if (encoder != null)
            {
                encoder.Dispose();
                encoder = null;
            }

            if (decoder != null)
            {
                decoder.Dispose();
                decoder = null;
            }
        }

        public void SetImage(byte[] image)
        {
            if (!initalized)
                return;

            if (image == null || image.Length == 0)
            {
                throw new ArgumentException("Invalid image");
            }


            byte[] resizedPixels = ImageUtility.GetPixelsResized(image, sideLength, out oldWidth, out oldHeight);
            byte[] inputTensorValues = new byte[sideLength * sideLength * 3];
            ImageUtility.GetResizedDimensions(oldWidth, oldHeight, sideLength, out newWidth, out newHeight);

            int p = 0;
            for (int i = 0; i < sideLength; i++)
            {
                for (int j = 0; j < sideLength; j++)
                {
                    int a = i * sideLength + j;
                    int b = sideLength * sideLength + i * sideLength + j;
                    int c = 2 * sideLength * sideLength + i * sideLength + j;

                    inputTensorValues[a] = resizedPixels[(p * 3) + 2];
                    inputTensorValues[b] = resizedPixels[(p * 3) + 1];
                    inputTensorValues[c] = resizedPixels[(p * 3)];

                    p++;
                }
            }
            DenseTensor<byte> inputTensor = new DenseTensor<byte>(inputTensorValues, new[] { 1, 3, sideLength, sideLength });
            using var results = encoder.Run(new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input", inputTensor) });
            imageEmbedding = results.First().AsTensor<float>().ToDenseTensor();
            SetImageInputs();
        }
        
        public void SetImage(string imgPath) { 
            if (!initalized)
                return;
            if (string.IsNullOrEmpty(imgPath) || !System.IO.File.Exists(imgPath))
            {
                return;
            }


            byte[] resizedPixels = ImageUtility.GetPixelsResized(imgPath, sideLength, out oldWidth, out oldHeight);
            ImageUtility.GetResizedDimensions(oldWidth, oldHeight, sideLength, out newWidth, out newHeight);
            byte[] inputTensorValues = new byte[sideLength * sideLength * 3];
            int p = 0;
            for (int i = 0; i < sideLength; i++)
            {
                for (int j = 0; j < sideLength; j++)
                {
                    int a = i * sideLength + j;
                    int b = sideLength * sideLength + i * sideLength + j;
                    int c = 2 * sideLength * sideLength + i * sideLength + j;

                    inputTensorValues[a] = resizedPixels[(p * 3) + 2];
                    inputTensorValues[b] = resizedPixels[(p * 3) + 1];
                    inputTensorValues[c] = resizedPixels[(p * 3)];

                    p++;
                }
            }
            DenseTensor<byte> inputTensor = new DenseTensor<byte>(inputTensorValues, new[] { 1, 3, sideLength, sideLength });
            using var results = encoder.Run(new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input", inputTensor) });
            imageEmbedding = results.First().AsTensor<float>().ToDenseTensor();
            SetImageInputs();
        }

        private void InitList() {

            DenseTensor<float> labelTensor = new(new[] { 1f }, new[] { 1, 1 });
            inputList.Add(NamedOnnxValue.CreateFromTensor("point_labels", labelTensor));

            DenseTensor<float> maskInput = new(new float[256 * 256], new[] { 1, 1, 256, 256 });
            inputList.Add(NamedOnnxValue.CreateFromTensor("mask_input", maskInput));

            DenseTensor<float> hasMask = new(new[] { 0f }, new[] { 1 });
            inputList.Add(NamedOnnxValue.CreateFromTensor("has_mask_input", hasMask));
        }

        private void SetImageInputs() {

            int index = inputList.FindIndex(x => x.Name == "orig_im_size");
            if (index != -1) {
                inputList.RemoveAt(index);
            }
            DenseTensor<float> originalSize = new(new[] { (float)oldHeight, (float)oldWidth }, new[] { 2 });
            inputList.Add(NamedOnnxValue.CreateFromTensor("orig_im_size", originalSize));

            index = inputList.FindIndex(x => x.Name == "image_embeddings");
            if (index != -1) {
                inputList.RemoveAt(index);
            }

            inputList.Add(NamedOnnxValue.CreateFromTensor("image_embeddings", imageEmbedding));
        }

       
    
        // Takes in mouse coordinates and returns mask of image
        public float[] GetPointMask(int xPoint, int yPoint)
        {
            if (!initalized)
            {
                return null;
            }

            xPoint = (int)(xPoint * ((float)newWidth / oldWidth));
            yPoint = (int)(yPoint * ((float)newHeight / oldHeight));


            int index = inputList.FindIndex(x => x.Name == "point_coords");
            if (index != -1) {
                inputList.RemoveAt(index);
            }

            DenseTensor<float> pointTensor = new(new[] {(float)xPoint, (float)yPoint }, new[] {1,1,2},false);
            inputList.Add(NamedOnnxValue.CreateFromTensor<float>("point_coords", pointTensor));

            using var results = decoder.Run(inputList);
            var masks = results.Where(x => x.Name == "masks").First().AsTensor<float>();
            float[] result = masks.ToArray();

            return result;

        }
    }
}