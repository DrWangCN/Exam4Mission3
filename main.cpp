#include "opencv2/opencv.hpp"
#include <iostream>
#include <filesystem>
#include <vector>

using namespace cv;
using namespace std;
using namespace filesystem;

int main() {
    // 读取行车图片和交通标志模板图片
    string archive_path = "./dataset/archive/";
    vector<string> archive_img;
    string template_path = "./dataset/template/";
    vector<string> template_img;
    for (const auto &entry : directory_iterator(archive_path)) {
        if (entry.path().extension() == ".jpg")
        {
            archive_img.push_back(entry.path().string());
        }
    }
    for (const auto &entry : directory_iterator(template_path)) {
        if (entry.path().extension() == ".jpg")
        {
            template_img.push_back(entry.path().string());
        }
    }
    
    for (int i = 0; i<template_img.size(); i++)
    {
        for (int j = 0; j<archive_img.size(); j++)
        {
            Mat img_scene = imread(archive_img[j], IMREAD_GRAYSCALE);
            Mat img_object = imread(template_img[i], IMREAD_GRAYSCALE);
            if (img_scene.empty() || img_object.empty()) {
                cout << "无法读取图片" << endl;
                return -1;
            }

            // 检测SIFT关键点和描述符
            Ptr<SIFT> detector = SIFT::create();
            vector<KeyPoint> keypoints_object, keypoints_scene;
            Mat descriptors_object, descriptors_scene;

            detector->detectAndCompute(img_object, noArray(), keypoints_object, descriptors_object);
            detector->detectAndCompute(img_scene, noArray(), keypoints_scene, descriptors_scene);

            // 使用FLANN匹配器进行特征匹配
            FlannBasedMatcher matcher;
            vector< vector<DMatch> > knn_matches;
            matcher.knnMatch(descriptors_object, descriptors_scene, knn_matches, 2);

            // 使用ratio test筛选好的匹配点
            const float ratio_thresh = 0.7f;
            vector<DMatch> good_matches;
            for (size_t i = 0; i < knn_matches.size(); i++) {
                if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
                    good_matches.push_back(knn_matches[i][0]);
                }
            }

            // 绘制匹配结果
            Mat img_matches;
            drawMatches(img_object, keypoints_object, img_scene, keypoints_scene, good_matches, img_matches, Scalar::all(-1),
                        Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            // imwrite("./matches/" + to_string(i+1) + " in " + to_string(j+1) + ".jpg", img_matches);
            cout << "good_matches: " << good_matches.size() << "  ";

            if (good_matches.size() > 0) {
                imwrite("./matches/" + to_string(i+1) + " in " + to_string(j+1) + ".jpg", img_matches);
            }

            if (good_matches.size() > 4) {
                // 获取关键点位置
                vector<Point2f> obj;
                vector<Point2f> scene;

                for (size_t i = 0; i < good_matches.size(); i++) {
                    // 获取物体的关键点位置
                    obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
                    // 获取场景的关键点位置
                    scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
                }

                // 计算单应性矩阵
                Mat H = findHomography(obj, scene, RANSAC);

                // 获取物体角点
                vector<Point2f> obj_corners(4);
                obj_corners[0] = Point2f(0, 0);
                obj_corners[1] = Point2f((float)img_object.cols, 0);
                obj_corners[2] = Point2f((float)img_object.cols, (float)img_object.rows);
                obj_corners[3] = Point2f(0, (float)img_object.rows);
                vector<Point2f> scene_corners(4);

                // 调试输出
                // cout << "obj_corners: " << obj_corners << endl;
                // cout << "H: " << H << endl;

                // 检查H的维度
                if (H.cols != 3 || H.rows != 3) {
                    cerr << "Error: H is not a 3x3 matrix" << endl;
                    continue;
                }

                // 将物体角点映射到场景角点
                perspectiveTransform(obj_corners, scene_corners, H);

                // 在场景图像中绘制边界
                line(img_matches, scene_corners[0] + Point2f((float)img_object.cols, 0),
                    scene_corners[1] + Point2f((float)img_object.cols, 0), Scalar(0, 255, 0), 4);
                line(img_matches, scene_corners[1] + Point2f((float)img_object.cols, 0),
                    scene_corners[2] + Point2f((float)img_object.cols, 0), Scalar(0, 255, 0), 4);
                line(img_matches, scene_corners[2] + Point2f((float)img_object.cols, 0),
                    scene_corners[3] + Point2f((float)img_object.cols, 0), Scalar(0, 255, 0), 4);
                line(img_matches, scene_corners[3] + Point2f((float)img_object.cols, 0),
                    scene_corners[0] + Point2f((float)img_object.cols, 0), Scalar(0, 255, 0), 4);

                // 显示最终结果
                imwrite("./output/" + to_string(i+1) + " in " + to_string(j+1) + ".jpg", img_matches);
                cout << "找到足够匹配点 " + to_string(i+1) + " in " + to_string(j+1) + ".jpg" << endl;
            } else {
                cout << "没有找到足够的匹配点" << endl;
            }
        }
    }
    
    cout << "Done" << endl;
    return 0;
}

