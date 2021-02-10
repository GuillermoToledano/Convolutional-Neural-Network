// CNN.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <opennn.h>
#include <opencv2/opencv.hpp>
#include "CIFAR.h"

//using namespace cv;
using namespace std;
using namespace OpenNN;
using namespace CIFAR10;

typedef unsigned char uchar;

int main()
{
    // Batch size
    const int Images = 20;
    // Strings
    string Path = "../batches-bin/data_batch_1.bin";

    char answer;
    // Store all image labels
    uint8_t Labels[Images];
    /* 
     Images:    Total of images to store
     ImageDim:  First dimension of the stored tensors
     ImageDim:  Second dimension of the stored tensors
     Channels:  Third dimension of the stored tensors
    */
    OpenNN::Tensor<uchar> Batch(Images, CIFAR10::ImageDim, CIFAR10::ImageDim, CIFAR10::Channels);

    CIFAR10::ReadBatch(Path, Labels, Batch, Images);
    cout << "Display images? (y/n) ";
    cin >> answer;
    cout << endl;
    if (answer == 'y')
    {
        for (size_t i = 0; i < Images; i++)
        {
            OpenNN::Tensor<uchar> Image = Batch.get_tensor(i);
            CIFAR10::DisplayImage(Image, Labels[i]);
            Image.~Tensor();
        }
    }    
    
    const Vector<size_t> Dimensions({3, 32, 32});
    const Vector<size_t> Filter({32, 5, 5});
    
    // CifarNet architecture
    OpenNN::NeuralNetwork CNN;
    /*
    // Convolutional Layer 1
    OpenNN::ConvolutionalLayer* ConvLayer_1 = new OpenNN::ConvolutionalLayer(Dimensions, Filter);
    CNN.add_layer(ConvLayer_1);
    const Vector<size_t> CL1_OutputDims = ConvLayer_1->get_outputs_dimensions();

    // Pooling Layer 1
    OpenNN::PoolingLayer* PoolLayer_1 = new OpenNN::PoolingLayer(CL1_OutputDims);
    CNN.add_layer(PoolLayer_1);
    const Vector<size_t> PL1_OutputDims = PoolLayer_1->get_outputs_dimensions();
    
    CNN.print_summary();
    */
    return 0;
}
