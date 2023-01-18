#include <iostream>
#include "planeDetect.h"
#include <glog/logging.h>
#include <pcl/io/auto_io.h>
#include <pcl/visualization/pcl_visualizer.h>

int main(int argc, char **argv) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    std::vector<planeFeature> planeParameters;
    int neighborNumber = 30, clusterThreshold = 50;
    double planeThreshold = 0.7, resolution = 0.5, planeClusterThreshold = 0.1, ransacParameter = 0.05;
    std::string lidarPath;

    if (argc > 6) {
        lidarPath = argv[1];
        resolution = std::stod(argv[2]);
        neighborNumber = std::stoi(argv[3]);
        planeThreshold = std::stoi(argv[4]);
        planeClusterThreshold = std::stod(argv[5]);
        clusterThreshold = std::stoi(argv[6]);
        ransacParameter = std::stod(argv[7]);
    }

    std::cout << "resolution: " << resolution << ", neighborNumber: " << neighborNumber << ", planeThreshold: "
              << planeThreshold << ", planeClusterThreshold: " << planeClusterThreshold << ", clusterThreshold: "
              << clusterThreshold << ", ransacParameter: " << ransacParameter << std::endl;

    pcl::io::load(lidarPath, *cloud);

    planeDetect planeDetect(resolution, neighborNumber, planeThreshold, planeClusterThreshold, clusterThreshold,
                            ransacParameter);
    planeDetect.setVisualization(true);
    planeDetect.pointSelect(cloud);
    planeDetect.cluster();
    planeDetect.getParameter(planeParameters);

    return 0;
}
