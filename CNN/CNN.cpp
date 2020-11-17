// CNN.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <opennn.h>
#include <opencv2/opencv.hpp>
#include "CIFAR.h"

using namespace cv;
using namespace std;
using namespace OpenNN;

int main()
{
    string path = "../batches-bin/data_batch_1.bin";
    int limit = 1;
    uint8_t labels[1];
    CIFAR10::ReadBatch(path, limit, labels);
    return 0;
}
