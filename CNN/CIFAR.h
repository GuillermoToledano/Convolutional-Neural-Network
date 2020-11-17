#pragma once
#ifndef CIFAR_H
#define CIFAR_H

using namespace std;

namespace CIFAR10 {
	const int ImageDim = 32;
	const int BatchSize = 10000;
	const int ImageSize = ImageDim * ImageDim;
	const int ImageBytes = ImageSize * 3;

	void ReadImage(std::unique_ptr<char[]>buff, int &labels) 
	{
		
	}

	void ReadBatch(const string path, int limit, uint8_t *Labels)
	{
		std::ifstream file;
		// Input, binary type, move pointer to end of file
		file.open(path, std::ios::in | std::ios::binary | std::ios::ate);
		if (!file) 
		{
			std::cout << "Error opening file!" << std::endl;
			return;
		}
		// Current position of the pointer
		auto file_size = file.tellg();
		// Create a pointer for dynamic memory
		std::unique_ptr<char[]> buffer(new char[file_size]);
		// Read the file
		file.seekg(0, std::ios::beg);
		file.read(buffer.get(), file_size);
		file.close();
		
		OpenNN::Matrix<std::uint8_t> RChannel(ImageDim, ImageDim, 0);
		OpenNN::Matrix<std::uint8_t> GChannel(ImageDim, ImageDim, 0);
		OpenNN::Matrix<std::uint8_t> BChannel(ImageDim, ImageDim, 0);
		OpenNN::Tensor<std::uint8_t> Image(ImageDim, ImageDim, 3);
		
		for (size_t i = 0; i < 1; i++)
		{
			Labels[i] = buffer[i * ImageBytes + 1];
			size_t x = 0;
			size_t y = 0;
			for (size_t j = 1; j < ImageSize + 1; j++)
			{
				RChannel(x, y) = buffer[i * ImageBytes + j];
				GChannel(x, y) = buffer[i * ImageBytes + j + 1025];
				BChannel(x, y) = buffer[i * ImageBytes + j + 2049];
				//cout << j << " " << buffer[i * ImageBytes + j] << " " << x << " " << y << " " << RChannel(x, y) << endl;
				y++;
				if (y % ImageDim == 0) {
					x++;
					y = 0;
				}
			}
			Image.add_matrix(RChannel);
			Image.add_matrix(GChannel);
			Image.add_matrix(BChannel);
		}		
	}
}

#endif // !CIFAR.h