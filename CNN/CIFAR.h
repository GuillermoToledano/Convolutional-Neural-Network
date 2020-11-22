#pragma once
#ifndef CIFAR_H
#define CIFAR_H

using namespace std;

typedef unsigned char uchar;

namespace CIFAR10 {
	const size_t Channels = 3;
	const size_t ImageDim = 32;
	const size_t BatchSize = 10000;
	const size_t ImageSize = ImageDim * ImageDim;
	const size_t ImageBytes = ImageSize * Channels;

	void DisplayImage(OpenNN::Tensor<uchar> Image);
	void ReadBatch(const string path, uint8_t* Labels, OpenNN::Tensor<uchar> Batch);

	void ReadBatch(const string path, uint8_t *Labels, OpenNN::Tensor<uchar> Batch)
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
		
		OpenNN::Matrix<uchar> RChannel(ImageDim, ImageDim, 1);
		OpenNN::Matrix<uchar> GChannel(ImageDim, ImageDim, 1);
		OpenNN::Matrix<uchar> BChannel(ImageDim, ImageDim, 1);

		/*OpenNN::Vector<size_t> Dimensions({ImageDim, ImageDim, Channels});
		OpenNN::Tensor<std::uint8_t> Image(Dimensions, 1);*/
		OpenNN::Tensor<uchar> Image(Channels, ImageDim, ImageDim);
		cout << Image.get_dimensions() << endl;

		for (uint16_t i = 0; i < 1; i++)
		{
			Labels[i] = buffer[i * ImageBytes + 1];
			size_t x = 0;
			size_t y = 0;
			for (uint16_t j = 1; j < ImageSize + 1; j++)
			{
				RChannel(x, y) = buffer[i * ImageBytes + j];
				GChannel(x, y) = buffer[i * ImageBytes + j + 1025];
				BChannel(x, y) = buffer[i * ImageBytes + j + 2049];
				// Rows & Cols
				y++;
				if (y % ImageDim == 0) {
					x++;
					y = 0;
				}
			}
			// Fill tensor
			Image.set_matrix(0, RChannel);
			Image.set_matrix(1, GChannel);
			Image.set_matrix(2, BChannel);
			// 
			//Batch.set_tensor(i, Image);
			//DisplayImage(Image);
		}		
	}

	void DisplayImage(OpenNN::Tensor<uchar> Image) {
		uchar Data[3072];
		uchar Dim = 32;
		const size_t rank = Image.get_dimension(2);
		const size_t rows = Image.get_dimension(0);
		const size_t cols = Image.get_dimension(1);

		size_t x = 0;
		for (size_t k = 0; k < rank; k++)
		{
			for (size_t i = 0; i < rows; i++)
			{
				for (size_t j = 0; j < cols; j++)
				{
					Data[x] = Image(i, j, k);
					x++;
				}
			}
		}	
		
		cv::Mat Img(ImageDim, ImageDim, CV_8UC3, Data);
		cv::imshow("CIFAR", Img);
		char key = cv::waitKey(0);
		if (key == 's')
		{
			imwrite("test.png", Img);
		}
	}
}

#endif // !CIFAR.h