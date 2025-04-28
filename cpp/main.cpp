#include "Fieldscale.h"
#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <16-bit_image> [out.png]\n";
        return 1;
    }
    const std::string inPath  = argv[1];
    const std::string outPath = (argc >= 3) ? argv[2] : "fieldscaled.png";

    cv::Mat input = cv::imread(inPath, cv::IMREAD_UNCHANGED);
    if (input.empty()) {
        std::cerr << "Could not read image: " << inPath << '\n';
        return 2;
    }
    if (input.type() != CV_16U && input.type() != CV_32S && input.type() != CV_64F) {
        std::cerr << "Expected single-channel 16-bit (or higher) image.\n";
        return 3;
    }

    Fieldscale fs;
    cv::Mat result = fs.process(input);

    // Save and display
    if (!cv::imwrite(outPath, result)) {
        std::cerr << "Failed to write: " << outPath << '\n';
    } else {
        std::cout << "Saved: " << outPath << '\n';
    }

    cv::imshow("Fieldscaled result", result);
    cv::waitKey(0);
    return 0;
}
