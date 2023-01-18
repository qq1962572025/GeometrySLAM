#ifndef PLANEDETECT_PLANEDETECT_H
#define PLANEDETECT_PLANEDETECT_H

#include <chrono>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>

class planeFeature {
public:
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr featureClouds_;
    Eigen::Vector4f planeParameter_, meanPosition_;
    double radius_;
    int id_, size_;

    planeFeature(int id = 0) : id_(id) {
        featureClouds_.reset(new pcl::PointCloud<pcl::PointXYZINormal>);
        radius_ = 0;
        size_ = 0;
        planeParameter_.setZero();
        meanPosition_.setZero();
    }


};

class planeDetect {

public:

    int neighborNumber_, clusterNumberThreshold_, pointNumber_;
    double planeThreshold_, planeClusterThreshold_, resolution_, ransacParameter_;

    bool useVisualization_;

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloudDS_;
    pcl::PointCloud<pcl::Normal>::Ptr cloudNormals_;
    std::vector<std::vector<int>> clusterLabels_, pointNeighbours_;
    pcl::search::KdTree<pcl::PointXYZI> search_;
    std::vector<double> planarity_;
    std::vector<Eigen::Vector3f> normals_, positions_;

    planeDetect(double resolution = 0.1, int neighborNumber = 30, double planeThreshold = 0.7,
                double planeClusterThreshold = 0.1,
                int clusterThreshold = 50, double ransacParameter = 0.05) :
            resolution_(resolution), neighborNumber_(neighborNumber), planeThreshold_(planeThreshold),
            clusterNumberThreshold_(clusterThreshold), planeClusterThreshold_(planeClusterThreshold),
            ransacParameter_(ransacParameter) {

        cloudDS_.reset(new pcl::PointCloud<pcl::PointXYZI>);
        cloudNormals_.reset(new pcl::PointCloud<pcl::Normal>);
        useVisualization_ = false;
    }

    void handleEigenValue(const Eigen::Vector3f &EigenValue, std::vector<int> &index) {
        if (EigenValue(0) >= EigenValue[1]) {
            if (EigenValue(1) >= EigenValue[2]) {
                std::vector<int>{0, 1, 2}.swap(index);
                return;
            }
            if (EigenValue[0] > EigenValue[2]) {
                std::vector<int>{0, 2, 1}.swap(index);
                return;
            } else {
                std::vector<int>{2, 0, 1}.swap(index);
                return;
            }
        } else {
            if (EigenValue(0) >= EigenValue[2]) {
                std::vector<int>{1, 0, 2}.swap(index);
                return;
            }
            if (EigenValue[1] > EigenValue[2]) {
                std::vector<int>{1, 2, 0}.swap(index);
                return;
            } else {
                std::vector<int>{2, 1, 0}.swap(index);
                return;
            }
        }
    }

    void setVisualization(bool value) {
        useVisualization_ = value;
    }

    void pointSelect(pcl::PointCloud<pcl::PointXYZI>::Ptr point) {
        pcl::VoxelGrid<pcl::PointXYZI> voxelGrid;
        voxelGrid.setLeafSize(resolution_, resolution_, resolution_);
        voxelGrid.setInputCloud(point);
        voxelGrid.filter(*cloudDS_);
        search_.setInputCloud(cloudDS_);

        pointNumber_ = cloudDS_->size();
        clusterLabels_.resize(0);
        pointNeighbours_.resize(0);
        normals_.resize(0);
        positions_.resize(0);
        planarity_.resize(pointNumber_);
        pointNeighbours_.resize(pointNumber_);
        cloudNormals_->resize(pointNumber_);

        std::vector<float> distances(neighborNumber_);
        std::vector<int> neighbours(neighborNumber_);
        for (int i_point = 0; i_point < cloudDS_->size(); i_point++) {
            search_.nearestKSearch(i_point, neighborNumber_, neighbours, distances);
            pointNeighbours_[i_point].swap(neighbours);
        }

        for (int i = 0; i < pointNumber_; i++) {
            Eigen::Vector4f norm;
            EIGEN_ALIGN16
            Eigen::Matrix3f covarianceMatrix;
            Eigen::Vector4f xyzCentroid;
            pcl::computeMeanAndCovarianceMatrix(*cloudDS_, pointNeighbours_[i], covarianceMatrix, xyzCentroid);

            Eigen::EigenSolver<Eigen::Matrix3f> es(covarianceMatrix);
            Eigen::Vector3f eigenValue = es.eigenvalues().real().array().abs();
            Eigen::Matrix3f eigenVector = es.eigenvectors().real();

            std::vector<int> index;
            handleEigenValue(eigenValue, index);

            norm.head<3>() = eigenVector.col(index[2]);
            norm[3] = -1 * norm.head<3>().dot(cloudDS_->points[i].getVector3fMap());

            double plane = (eigenValue(index[1]) - eigenValue.minCoeff()) / eigenValue.maxCoeff();

            cloudNormals_->at(i).getNormalVector4fMap() = norm;
            planarity_[i] = plane;
        }

    }

    void cluster() {

        std::vector<int> labels(pointNumber_, 0);

        auto t1 = std::chrono::steady_clock::now();
        int seed = 0, labelNumPlane_(0);

        while (seed + 1 < pointNumber_) {

            while (seed + 1 < pointNumber_ && (labels[seed] > 0 || planarity_[seed] < planeThreshold_))
                seed++;
            labelNumPlane_++;

            std::queue<int> seeds;
            std::vector<int> clusterLabel;
            seeds.push(seed);
            labels[seed] = labelNumPlane_;
            Eigen::Vector3f meanNormal = Eigen::Vector3f::Zero(), meanPosition = Eigen::Vector3f::Zero();
            while (!seeds.empty()) {
                auto point = cloudNormals_->at(seeds.front()).getNormalVector3fMap();
                meanNormal += point;
                meanPosition += cloudDS_->at(seeds.front()).getVector3fMap();
                for (auto iter: pointNeighbours_[seeds.front()])
                    if (planarity_[iter] > planeThreshold_ && labels[iter] == 0) {
                        auto nei = cloudNormals_->at(iter).getNormalVector3fMap();
                        double errorNormal;
                        errorNormal = point.cross(nei).norm();
                        if (errorNormal < planeClusterThreshold_) {
                            labels[iter] = labelNumPlane_;
                            seeds.push(iter);
                            clusterLabel.push_back(iter);
                        }
                    }
                seeds.pop();
            }

            meanNormal.normalize();

            if (clusterLabel.size() > clusterNumberThreshold_) {
                meanPosition /= clusterLabel.size();
                clusterLabels_.push_back(clusterLabel);
                normals_.push_back(meanNormal);
                positions_.push_back(meanPosition);
            }

        }

        std::cout << "Plane size: " << clusterLabels_.size() << std::endl;

    }

    void getParameter(std::vector<planeFeature> &planeFeatures) {

        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::SACSegmentation<pcl::PointXYZI> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(ransacParameter_);

        pcl::visualization::PCLVisualizer viewer("plane");

        for (int i = 0; i < clusterLabels_.size(); i++) {

            pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
            for (auto iter: clusterLabels_[i])
                cloud->push_back(cloudDS_->at(iter));
            seg.setInputCloud(cloud);
            seg.segment(*inliers, *coefficients);

            planeFeature planeFeatureA(i);
            planeFeatureA.size_ = inliers->indices.size();
            planeFeatureA.planeParameter_ = Eigen::Vector4f(coefficients->values[0], coefficients->values[1],
                                                            coefficients->values[2], coefficients->values[3]);
            for (auto iter: inliers->indices) {
                pcl::PointXYZINormal point;
                point.getVector3fMap() = cloud->at(iter).getVector3fMap();
                point.getNormalVector4fMap() = cloudNormals_->at(clusterLabels_[i][iter]).getNormalVector4fMap();
                planeFeatureA.meanPosition_ += point.getVector4fMap();
                if ((point.getVector3fMap() - positions_[i]).norm() > planeFeatureA.radius_)
                    planeFeatureA.radius_ = (point.getVector3fMap() - positions_[i]).norm();
            }
            planeFeatureA.meanPosition_ /= planeFeatureA.size_;

            if (useVisualization_) {
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudColor(new pcl::PointCloud<pcl::PointXYZRGB>);
                int r, g, b;
                r = rand() % 255;
                g = rand() % 255;
                b = rand() % 255;
                for (auto iter: inliers->indices) {
                    pcl::PointXYZRGB point;
                    point.getVector3fMap() = cloud->at(iter).getVector3fMap();
                    point.r = r;
                    point.g = g;
                    point.b = b;
                    cloudColor->push_back(point);
                }
                viewer.addPointCloud(cloudColor, std::to_string(i));
            }


        }

        if (useVisualization_)
            viewer.spin();

    }

};

#endif //PLANEDETECT_PLANEDETECT_H
