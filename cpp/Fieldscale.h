#ifndef FIELDSCALE_H
#define FIELDSCALE_H

#include <opencv2/opencv.hpp>

class Fieldscale {
public:
    Fieldscale(double maxDiff = 400.0, double minDiff = 400.0, 
               int iterations = 7, double gamma = 1.5, 
               bool clahe = true, bool video = false);

    cv::Mat process(const cv::Mat& input);

private:
    double maxDiff_;
    double minDiff_;
    int iterations_;
    double gamma_;
    bool clahe_;
    bool video_;

    cv::Mat prevMinField_;
    cv::Mat prevMaxField_;

    cv::Mat gridwiseMin(const cv::Mat& img, const cv::Size& grid);
    cv::Mat gridwiseMax(const cv::Mat& img, const cv::Size& grid);
    cv::Mat localExtremaSuppression(const cv::Mat& grid, int distance, double threshold, const std::string& extrema);
    cv::Mat messagePassing(const cv::Mat& grid, const std::string& direction);
    cv::Mat rescaleImageWithFields(const cv::Mat& image, const cv::Mat& minField, const cv::Mat& maxField);
};

#endif
