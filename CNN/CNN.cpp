// CNN.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <opennn.h>
#include <opencv2/opencv.hpp>
#include "CIFAR.h"

using namespace cv;
using namespace std;
using namespace OpenNN;
using namespace CIFAR10;

typedef unsigned char uchar;

int main()
{
    string Path = "../batches-bin/data_batch_1.bin";

    uint8_t Labels[1];
    OpenNN::Tensor<uchar> Batch;
    CIFAR10::ReadBatch(Path, Labels, Batch);


    return 0;
}
