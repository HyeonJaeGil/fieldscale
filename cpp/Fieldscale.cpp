/*
Source code for the paper:

Fieldscale: Locality-Aware Field-based Adaptive Rescaling for Thermal Infrared Image
Hyeonjae Gil, Myeon-Hwan Jeon, and Ayoung Kim

Please cite the paper if you use this code.

@article{gil2024fieldscale,
  title={Fieldscale: Locality-Aware Field-based Adaptive Rescaling for Thermal Infrared Image},
  author={Gil, Hyeonjae and Jeon, Myung-Hwan and Kim, Ayoung},
  journal={IEEE Robotics and Automation Letters},
  year={2024},
  publisher={IEEE}
}

Original author: Hyeonjae Gil
Author email: h.gil@snu.ac.kr
*/

#include "Fieldscale.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <numeric>

using namespace cv;
using std::vector;


Fieldscale::Fieldscale(double maxDiff, double minDiff, int iterations,
                       double gamma, bool clahe, bool video)
    : maxDiff_(maxDiff), minDiff_(minDiff), iterations_(iterations),
      gamma_(gamma), clahe_(clahe), video_(video) {}


Mat Fieldscale::gridwiseMin(const Mat &img, const Size &grid) {
    Mat output(grid, CV_64F);
    Size patch(img.cols / grid.width, img.rows / grid.height);
    for (int y = 0; y < grid.height; ++y) {
        for (int x = 0; x < grid.width; ++x) {
            Rect roi(x * patch.width, y * patch.height, patch.width, patch.height);
            double minVal;
            minMaxLoc(img(roi), &minVal, nullptr);
            output.at<double>(y, x) = minVal;
        }
    }
    return output;
}

Mat Fieldscale::gridwiseMax(const Mat &img, const Size &grid) {
    Mat output(grid, CV_64F);
    Size patch(img.cols / grid.width, img.rows / grid.height);
    for (int y = 0; y < grid.height; ++y) {
        for (int x = 0; x < grid.width; ++x) {
            Rect roi(x * patch.width, y * patch.height, patch.width, patch.height);
            double maxVal;
            minMaxLoc(img(roi), nullptr, &maxVal);
            output.at<double>(y, x) = maxVal;
        }
    }
    return output;
}


Mat Fieldscale::localExtremaSuppression(const Mat &grid, int distance,
                                        double threshold,
                                        const std::string &extrema) {
    CV_Assert(extrema == "max" || extrema == "min");
    Mat result = grid.clone();

    for (int y = 0; y < grid.rows; ++y) {
        for (int x = 0; x < grid.cols; ++x) {
            vector<double> neighbors;
            for (int dy = -distance; dy <= distance; ++dy) {
                for (int dx = -distance; dx <= distance; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    int ny = y + dy, nx = x + dx;
                    if (ny >= 0 && ny < grid.rows && nx >= 0 && nx < grid.cols)
                        neighbors.push_back(grid.at<double>(ny, nx));
                }
            }
            if (neighbors.empty()) continue;

            // manual mean
            double meanVal = std::accumulate(neighbors.begin(), neighbors.end(), 0.0) /
                              static_cast<double>(neighbors.size());
            double current = grid.at<double>(y, x);

            if (extrema == "max") {
                double neighborMax = *std::max_element(neighbors.begin(), neighbors.end());
                if (current >= neighborMax && (current - meanVal) > threshold)
                    result.at<double>(y, x) = meanVal + threshold;
            } else { // min
                double neighborMin = *std::min_element(neighbors.begin(), neighbors.end());
                if (current <= neighborMin && (meanVal - current) > threshold)
                    result.at<double>(y, x) = meanVal - threshold;
            }
        }
    }
    return result;
}

Mat Fieldscale::messagePassing(const Mat &grid, const std::string &direction) {
    CV_Assert(direction == "increase" || direction == "decrease");
    Mat newGrid = grid.clone();

    for (int y = 0; y < grid.rows; ++y) {
        for (int x = 0; x < grid.cols; ++x) {
            vector<double> neighbors;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    int ny = y + dy, nx = x + dx;
                    if (ny >= 0 && ny < grid.rows && nx >= 0 && nx < grid.cols)
                        neighbors.push_back(grid.at<double>(ny, nx));
                }
            }

            double val = grid.at<double>(y, x);
            if (neighbors.empty()) {
                newGrid.at<double>(y, x) = val;
                continue;
            }

            double sum = std::accumulate(neighbors.begin(), neighbors.end(), 0.0) + val; // include self
            double meanVal = sum / static_cast<double>(neighbors.size() + 1);

            double bigger = std::max(meanVal, val);
            double smaller = std::min(meanVal, val);
            newGrid.at<double>(y, x) = (direction == "increase") ? bigger : smaller;
        }
    }
    return newGrid;
}

Mat Fieldscale::rescaleImageWithFields(const Mat &image, const Mat &minField,
                                       const Mat &maxField) {
    CV_Assert(image.size() == minField.size() && image.size() == maxField.size());

    Mat floatImg;
    image.convertTo(floatImg, CV_64F);

    Mat minF, maxF;
    min(minField, maxField, minF);
    max(minField, maxField, maxF);

    Mat clipped;
    clipped = max(floatImg, minF);
    clipped = min(clipped, maxF);

    Mat denom = maxF - minF;
    Mat mask = denom < 1e-6;
    denom.setTo(1.0, mask);

    Mat normalized;
    normalized = (clipped - minF).mul(1.0 / denom) * 255.0;
    normalized.convertTo(normalized, CV_8U);
    return normalized;
}

Mat Fieldscale::process(const Mat &input) {
    CV_Assert(!input.empty());

    Mat input64;
    input.convertTo(input64, CV_64F);

    Mat minGrid = gridwiseMin(input64, {8, 8});
    Mat maxGrid = gridwiseMax(input64, {8, 8});

    maxGrid = localExtremaSuppression(maxGrid, 2, maxDiff_, "max");
    maxGrid = localExtremaSuppression(maxGrid, 2, minDiff_, "min");

    for (int i = 0; i < iterations_; ++i) {
        minGrid = messagePassing(minGrid, "decrease");
        maxGrid = messagePassing(maxGrid, "increase");
    }

    Mat minField, maxField;
    resize(minGrid, minField, input.size(), 0, 0, INTER_LINEAR);
    resize(maxGrid, maxField, input.size(), 0, 0, INTER_LINEAR);

    if (video_ && !prevMinField_.empty()) {
        minField = 0.1 * minField + 0.9 * prevMinField_;
        maxField = 0.1 * maxField + 0.9 * prevMaxField_;
    }
    prevMinField_ = minField.clone();
    prevMaxField_ = maxField.clone();

    Mat result = rescaleImageWithFields(input, minField, maxField);

    if (gamma_ > 0) {
        result.convertTo(result, CV_32F, 1.0 / 255.0);
        pow(result, gamma_, result);
        result *= 255.0f;
        result.convertTo(result, CV_8U);
    }

    if (clahe_) {
        Ptr<CLAHE> claheObj = createCLAHE(2.0, Size(8, 8));
        claheObj->apply(result, result);
    }

    return result;
}
