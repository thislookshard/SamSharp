using System;
using System.Collections.Generic;
using System.Drawing.Imaging;
using System.Drawing;

namespace SamSharp
{
    internal class ImageUtility
    {
        
        public static void GetResizedDimensions(int width, int height, int sideLength, out int newWidth, out int newHeight)
        {
            double scale = sideLength * 1.0 / Math.Max(width, height);
            newHeight = (int)(height * scale + 0.5);
            newWidth = (int)(width * scale+0.5);

        }

        public static byte[] GetOriginalPixels(byte[] imgBytes)
        {
            using var stream = new MemoryStream(imgBytes);
            using Bitmap bitmap = new Bitmap(stream);

            BitmapData bmp = bitmap.LockBits(new Rectangle(0, 0, bitmap.Width, bitmap.Height), ImageLockMode.ReadWrite, PixelFormat.Format24bppRgb);
            byte[] rgbValues = new byte[bmp.Height * bmp.Width * 3];
            System.Runtime.InteropServices.Marshal.Copy(bmp.Scan0, rgbValues, 0, rgbValues.Length);
            bitmap.UnlockBits(bmp);
            return rgbValues;

        }

        public static byte[] GetPixelsResized(byte[] imgBytes, int sideLength, out int oldWith, out int oldHeight)
        {
            using var stream = new MemoryStream(imgBytes);

            using Bitmap bitmap = new Bitmap(stream);
            oldHeight = bitmap.Height;
            oldWith = bitmap.Width;

            GetResizedDimensions(bitmap.Width, bitmap.Height, sideLength, out int resizedWidth, out int resizedHeight);

            using Bitmap bmp = new Bitmap(resizedWidth, resizedHeight, PixelFormat.Format32bppRgb);
            using Graphics g = Graphics.FromImage(bmp);
            g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
            g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.HighQuality;
            g.PixelOffsetMode = System.Drawing.Drawing2D.PixelOffsetMode.HighQuality;
            g.CompositingQuality = System.Drawing.Drawing2D.CompositingQuality.HighQuality;
            g.DrawImage(bitmap, 0, 0, resizedWidth, resizedHeight);


            using Bitmap paddedMap = new Bitmap(sideLength, sideLength, PixelFormat.Format24bppRgb);
            using var pg = Graphics.FromImage(paddedMap);
            pg.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
            pg.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.HighQuality;
            pg.PixelOffsetMode = System.Drawing.Drawing2D.PixelOffsetMode.HighQuality;
            pg.CompositingQuality = System.Drawing.Drawing2D.CompositingQuality.HighQuality;
            pg.DrawImage(bmp, 0, 0);


            Rectangle rect = new Rectangle(0, 0, paddedMap.Width, paddedMap.Height);
            BitmapData bmpData = paddedMap.LockBits(rect, ImageLockMode.ReadWrite, PixelFormat.Format24bppRgb);
            int byteSize = Math.Abs(bmpData.Stride) * bmpData.Height;
            byte[] rgbValues = new byte[byteSize];
            System.Runtime.InteropServices.Marshal.Copy(bmpData.Scan0, rgbValues, 0, byteSize);
            paddedMap.UnlockBits(bmpData);



            return rgbValues;
        }
        public static byte[] GetPixelsResized(string path, int sideLength, out int oldWith, out int oldHeight)
        {
            using Bitmap bitmap = new Bitmap(path);
            oldHeight = bitmap.Height;
            oldWith = bitmap.Width;

            GetResizedDimensions(bitmap.Width, bitmap.Height, sideLength, out int resizedWidth, out int resizedHeight);

            using Bitmap bmp = new Bitmap(resizedWidth, resizedHeight, PixelFormat.Format32bppRgb);
            using Graphics g = Graphics.FromImage(bmp);
            g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
            g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.HighQuality;
            g.PixelOffsetMode = System.Drawing.Drawing2D.PixelOffsetMode.HighQuality;
            g.CompositingQuality = System.Drawing.Drawing2D.CompositingQuality.HighQuality;
            g.DrawImage(bitmap, 0, 0, resizedWidth, resizedHeight);


            using Bitmap paddedMap = new Bitmap(sideLength, sideLength, PixelFormat.Format24bppRgb);
            using var pg = Graphics.FromImage(paddedMap);
            pg.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
            pg.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.HighQuality;
            pg.PixelOffsetMode = System.Drawing.Drawing2D.PixelOffsetMode.HighQuality;
            pg.CompositingQuality = System.Drawing.Drawing2D.CompositingQuality.HighQuality;
            pg.DrawImage(bmp, 0, 0);

            string savePath = System.IO.Path.GetDirectoryName(path) + "//"+System.IO.Path.GetFileNameWithoutExtension(path) + "_resized_padded.jpg";
            paddedMap.Save(savePath);
            Rectangle rect = new Rectangle(0, 0, paddedMap.Width, paddedMap.Height);
            BitmapData bmpData = paddedMap.LockBits(rect, ImageLockMode.ReadWrite, PixelFormat.Format24bppRgb);
            int byteSize = Math.Abs(bmpData.Stride) * bmpData.Height;
            byte[] rgbValues = new byte[byteSize];
            System.Runtime.InteropServices.Marshal.Copy(bmpData.Scan0, rgbValues, 0, byteSize);
            paddedMap.UnlockBits(bmpData);



            return rgbValues;
        }

        public static void SaveImage(byte[] rgbValues,int width, int height, string path)
        {
            using var stream = new FileStream(path, FileMode.OpenOrCreate);
            using Bitmap bitmap = new Bitmap(width, height, PixelFormat.Format24bppRgb);

            
            BitmapData bmpData = bitmap.LockBits(new Rectangle(0, 0, bitmap.Width, bitmap.Height), ImageLockMode.ReadWrite, PixelFormat.Format24bppRgb);

            System.Runtime.InteropServices.Marshal.Copy(rgbValues, 0, bmpData.Scan0, rgbValues.Length);
            bitmap.UnlockBits(bmpData);
            bitmap.Save(stream, ImageFormat.Jpeg);
        }
    }
}
