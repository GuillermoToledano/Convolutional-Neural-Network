#pragma once
#ifndef CIFAR_H
#define CIFAR_H

//#include <opennn.h>
//#include <opencv2/opencv.hpp>

typedef unsigned char uchar;

namespace CIFAR10 {

	void DisplayImage(OpenNN::Tensor<uchar> Image, uchar Label);
	void ReadBatch(const string path, uint8_t* Labels, OpenNN::Tensor<uchar> &Batch, int BatchSize);

	const size_t Channels = 3;
	const size_t ImageDim = 32;
	const size_t ImageSize = ImageDim * ImageDim;
	const size_t ImageBytes = ImageSize * Channels;
	const string Strs[] = {"airplane", "automobile", "bird", "cat", "deer", "dog", 
		                   "frog", "horse", "ship", "truck"};
	/// <summary>
	/// 
	/// </summary>
	/// <param name="path"></param>
	/// <param name="Labels"></param>
	/// <param name="Batch"></param>
	/// <param name="BatchSize"></param>
	void ReadBatch(const string path, uint8_t *Labels, OpenNN::Tensor<uchar> &Batch, int BatchSize)
	{
		std::ifstream file;
		file.open(path, std::ios::in | std::ios::binary | std::ios::ate);
		if (!file) 
		{
			std::cout << "Error opening file!" << std::endl;
			return;
		}
		auto file_size = file.tellg();
		std::unique_ptr<char[]> buffer(new char[file_size]);
		file.seekg(0, std::ios::beg);
		file.read(buffer.get(), file_size);
		file.close();		
		
		// This loop control the number of images to read
		for (uint16_t i = 0; i < BatchSize; i++)
		{
			Labels[i] = buffer[i * (ImageBytes + 1)];
			OpenNN::Matrix<uchar> RChannel(ImageDim, ImageDim, 0);
			OpenNN::Matrix<uchar> GChannel(ImageDim, ImageDim, 0);
			OpenNN::Matrix<uchar> BChannel(ImageDim, ImageDim, 0);
			OpenNN::Tensor<uchar> Image(ImageDim, ImageDim, Channels);
			
			size_t x = 0;
			size_t y = 0;
			// This loop reads each image by channel
			for (uint16_t j = 1; j < ImageSize + 1; j++)
			{
				// Improve using a vector? for-loop need to use ImageBytes instead
				// OpenNN::Vector<uchar> Data(ImageBytes);
				// Data.at(j-1) = buffer[i * ImageBytes + j];
				RChannel(x, y) = buffer[i * (ImageBytes + 1) + j];
				GChannel(x, y) = buffer[i * (ImageBytes + 1) + j + 1024];
				BChannel(x, y) = buffer[i * (ImageBytes + 1) + j + 2048];
				// Control rows and cols
				x++;
				if (ImageDim == x) {
					y++;
					x = 0;
				}
			}
			// Fill tensor
			Image.set_matrix(0, RChannel);
			Image.set_matrix(1, GChannel);
			Image.set_matrix(2, BChannel);
			// If using vector, look for embed method in Tensor class
			// Image.embed(0, Data);
			// Add Image tensor to Batch tensor
			Batch.set_tensor(i, Image);
		}		
	}
	/// <summary>
	/// Creates a window and display an image.
	/// </summary>
	/// <param name="Image">An tensor that conteins the image's data with dimensions (32, 32, 3)</param>
	/// <param name="Label">A value that represent the image's class</param>
	void DisplayImage(OpenNN::Tensor<uchar> Image, uchar Label)
	{
		uchar Data[3072]{};
		for (size_t i = 0; i < ImageSize; i++)
		{
			for (size_t j = 0; j < Channels; j++)
			{
				Data[i * Channels + j] = Image[j * ImageSize + i];
			}
		}
		cv::Mat Img(ImageDim, ImageDim, CV_8UC3, Data, ImageDim * Channels);
		cv::cvtColor(Img, Img, cv::COLOR_RGB2BGR, Channels);
		switch (Label)
		{
		case 0:
			cout << Strs[0] << endl;
			break;
		case 1:
			cout << Strs[1] << endl;
			break;
		case 2:
			cout << Strs[2] << endl;
			break;
		case 3:
			cout << Strs[3] << endl;
			break;
		case 4:
			cout << Strs[4] << endl;
			break;
		case 5:
			cout << Strs[5] << endl;
			break;
		case 6:
			cout << Strs[6] << endl;
			break;
		case 7:
			cout << Strs[7] << endl;
			break;
		case 8:
			cout << Strs[8] << endl;
			break;
		case 9:
			cout << Strs[9] << endl;
			break;
		}
		cv::namedWindow("CIFAR-10 Sample", cv::WINDOW_FREERATIO | cv::WINDOW_GUI_EXPANDED);
		cv::imshow("CIFAR-10 Sample", Img);
		cv::waitKey(0);
		cv::destroyWindow("CIFAR-10 Sample");
	}
}

#endif // !CIFAR.h