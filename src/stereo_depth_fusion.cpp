#include <stdio.h>
#include <ctime>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <dirent.h>
#include "stereo_sdf/SGMStereo.h"
#include "stereo_sdf/utils.h"

using namespace std;

/* -------------------------- SET THESE PARAMETERS --------------------------- */
/* stereo options */
#define STEREO_FULL_PIPELINE -1
#define STEREO_LEFT_ONLY      0

/* cost volume update options */
#define VOLUME_UPDATE_CROSS   4
#define VOLUME_UPDATE_NAIVE   3
#define VOLUME_UPDATE_DIFFB   2
#define VOLUME_UPDATE_NEIGH   1
#define VOLUME_UPDATE_NONE   -1

/* image save flag */
#define SAVE_IMAGES           1

/* ground truth sampling option */
double SAMPLING_FRACTION   = 0.50;

/* disparity scaling factor (256 for KITTI) */
double SCALING_FACTOR      = 256.0;

/* input and output directories */
std::string repo_dir = "/home/mm/stereo_sparse_depth_fusion-master/";
std::string left_image_uri = repo_dir + "imgs/stereo_left.png";
std::string right_image_uri = repo_dir + "imgs/stereo_right.png";
std::string left_depth_uri = repo_dir + "imgs/gt_disparity.png";
std::string save_dir = repo_dir + "results/";
/* --------------------------------------------------------------------------- */

void SemiGlobalMatching(const cv::Mat &leftImage,
                        const cv::Mat &rightImage,
                        cv::Mat &dispImage,                //计算之后的深度图存储的地方
                        int STEREO_PIPELINE_MODE,
                        const std::string cameraParamFile, //相机参数
                        cv::Mat depthImage,                //预先准备的深度图
                        cv::Mat weightImg,                 //权重存储
                        int FUSE_FLAG)
{
    //定义两个png类变量用以接收左右图像
    png::image<png::rgb_pixel> leftImageSGM, rightImageSGM;
    //使用Utils类
    Utils utilities;
    //将Mat类的图像转化成PNG类
    utilities.convertCVMatToPNG(leftImage, leftImageSGM);
    utilities.convertCVMatToPNG(rightImage, rightImageSGM);
    //测量左图像的宽和高
    size_t width = leftImageSGM.get_width();
    size_t height = leftImageSGM.get_height();
    //检测左右图像尺寸是否一致
    if (width != rightImageSGM.get_width() ||
        height != rightImageSGM.get_height())
    {
        dispImage = cv::Mat1w();
        return;
    }
    //为深度图像分配所需的内存空间
    float* dispImageFloat = (float*)malloc(width * height * sizeof(float));
    //使用SGMStereo类
    SGMStereo sgm;
    //将彩色的左图，转化成灰度图
    cv::Mat leftImageGray;
    cv::cvtColor(leftImage, leftImageGray, CV_RGB2GRAY);
    //开始计算
    sgm.compute(leftImageSGM,    //传入左图
        rightImageSGM,           //传入右图
        dispImageFloat,          //用来接收计算结果-深度图
        STEREO_PIPELINE_MODE,    //管道
        cameraParamFile,         //相机参数
        depthImage,              //预先准备的深度图
        FUSE_FLAG,               //判断是否融合
        leftImageGray,           //传入左图的灰度图
        weightImg);              //权重图
    //得到深度图像
    dispImage = utilities.convertFloatToCVMat(width, height, dispImageFloat);
    //释放内存
    free(dispImageFloat);
}

void displayMinMax(cv::Mat array)
{
    double min, max;
    cv::minMaxLoc(array, &min, &max);
    std::cout << "Minimum: " << min << " | Maximum: " << max << std::endl;
}

int main()
{
    Utils utilities;
    /* -------------------------读入图像read input images---------------------------------- */
    std::cout << "DATA DETAILS: " << std::endl;
    std::cout << "--- Left Image: " << left_image_uri << std::endl;
    std::cout << "--- Right Image: " << right_image_uri << std::endl;
    std::cout << "--- Disparity: (GT) " << left_depth_uri << std::endl;

    std::vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(0);

    cv::Mat left_image_clr = cv::imread(left_image_uri);
    cv::Mat right_image_clr = cv::imread(right_image_uri);

    cv::Mat_<double> disp_image = cv::imread(left_depth_uri,
                                             cv::IMREAD_ANYDEPTH);
    disp_image = disp_image / SCALING_FACTOR;
    std::cout << "{read status:} successfully read input images.." << std::endl;
    /* -------------------------------------------------------------------------------- */


    /* EVALUATION A: SEMI GLOBAL MATCHING */
    std::cout << "\n{EVALUATION A:} -- SEMI GLOBAL MATCHING -- " << std::endl;
    cv::Mat disparity_image;
    SemiGlobalMatching(left_image_clr,
                       right_image_clr,
                       disparity_image,
                       STEREO_LEFT_ONLY,
                       "no_params_needed",
                       cv::Mat(),
                       cv::Mat(),
                       VOLUME_UPDATE_NONE);
    if (SAVE_IMAGES) {
        std::string save_file_name = + "sgm_default.png";
        std::string save_url = save_dir + save_file_name;
        std::cout << "{SGM} saving image to: " << save_url << std::endl;
        cv::imwrite(save_url, disparity_image, compression_params);
    }

    cv::Mat_<double> disp_SGM = disparity_image / SCALING_FACTOR;
    /* evaluate SGM */
    cv::Mat_<double> error_image_sgm;
    double average_error_sgm;
    cv::Mat sample_mask_sgm = cv::Mat::zeros(disp_SGM.rows,
                                             disp_SGM.cols,
                                             CV_32FC1);
    utilities.calculateAccuracy(disp_SGM,
                                disp_image,
                                average_error_sgm,
                                error_image_sgm,
                                sample_mask_sgm);
    std::cout << "{SGM} avg error: " << average_error_sgm << std::endl;


// 特殊处理，为融合算法准备一张深度图
    cv::Mat sample_mask;
    utilities.generateRandomSamplingMask(disp_image,
                                         sample_mask,
                                         SAMPLING_FRACTION);
    if (SAVE_IMAGES) {
        std::string save_file_name = "sparse_mask.png";
        std::string save_url = save_dir + save_file_name;
        std::cout << "{MASK} saving image to: " << save_url << std::endl;
        cv::imwrite(save_url, sample_mask, compression_params);
    }
    cv::Mat masked_depth;
    disp_image.copyTo(masked_depth, sample_mask);//将提供的激光深度图disp_image复制给一个附加掩图sample_mask的masked_depth

    /* EVALUATION B: USE NAIVE FUSION METHOD*/
    std::cout << "\n{EVALUATION B:} -- NAIVE LIDAR FUSION -- " << std::endl;
    cv::Mat disparity_image_sl_naive;
    SemiGlobalMatching(left_image_clr,
                       right_image_clr,
                       disparity_image_sl_naive,
                       STEREO_LEFT_ONLY,
                       "no_params_needed",
                       masked_depth,           //与上一步SGM算法相差了此处的masked_depth参数，这就是激光深度图，用来做融合用
                       cv::Mat(),
                       VOLUME_UPDATE_NAIVE);
    if (SAVE_IMAGES) {
        std::string save_file_name = "fuse_naive.png";
        std::string save_url = save_dir + save_file_name;
        std::cout << "{Naive Fusion} saving image to: " << save_url << std::endl;
        cv::imwrite(save_url, disparity_image_sl_naive, compression_params);
    }
    cv::Mat_<double> disp_NF = disparity_image_sl_naive / SCALING_FACTOR; //最终的深度图
    /* evaluate naive fusion */
    cv::Mat_<double> error_image_nf;
    double average_error_nf;
    utilities.calculateAccuracy(disp_NF,
                                disp_image,
                                average_error_nf,
                                error_image_nf, sample_mask);
    std::cout << "{NAIVE FUSION} avg error: " << average_error_nf << std::endl;
 
    /*EVALUATION C: USE DIFFUSION BASED METHOD */
    std::cout << "\n{EVALUATION C:} -- DIFFUSION BASED -- " << std::endl;
    cv::Mat disparity_image_db;
    SemiGlobalMatching(left_image_clr,
                       right_image_clr,
                       disparity_image_db,
                       STEREO_LEFT_ONLY,
                       "no_params_needed",
                       masked_depth,
                       cv::Mat(),
                       VOLUME_UPDATE_DIFFB);
    if (SAVE_IMAGES) {
        std::string save_file_name = "fuse_diffusionbased.png";
        std::string save_url = save_dir + save_file_name;
        std::cout << "{DB} saving image to: " << save_url << std::endl;
        cv::imwrite(save_url, disparity_image_db, compression_params);
    }
    cv::Mat_<double> disp_DB = disparity_image_db / SCALING_FACTOR;
    /* evaluate diffusion based confidence propagation method */
    cv::Mat_<double> error_image_db;
    double average_error_db;
    utilities.calculateAccuracy(disp_DB,
                                disp_image,
                                average_error_db,
                                error_image_db,
                                sample_mask);
    std::cout << "{DB} avg error: " << average_error_db << std::endl;

    /*EVALUATION D: USE NEIGHBORHOOD SUPPORT METHOD */
    std::cout << "\n{EVALUATION D:} -- NEIGHBORHOOD SUPPORT -- " << std::endl;
    cv::Mat disparity_image_ns;
    SemiGlobalMatching(left_image_clr,
                       right_image_clr,
                       disparity_image_ns,
                       STEREO_LEFT_ONLY,
                       "no_params_needed",
                       masked_depth,
                       cv::Mat(),
                       VOLUME_UPDATE_NEIGH);
    if (SAVE_IMAGES) {
        std::string save_file_name = "fuse_neighborhoodsupport.png";
        std::string save_url = save_dir + save_file_name;
        std::cout << "{NS} saving image to: " << save_url << std::endl;
        cv::imwrite(save_url, disparity_image_ns, compression_params);
    }
    cv::Mat_<double> disp_NS = disparity_image_ns / SCALING_FACTOR;
    /* evaluate neighborhood support method */
    cv::Mat_<double> error_image_ns;
    double average_error_ns;
    utilities.calculateAccuracy(disp_NS,
                                disp_image,
                                average_error_ns,
                                error_image_ns,
                                sample_mask);
    std::cout << "{NS} avg error: " << average_error_ns << std::endl;

/*EVALUATION E: USE CROSS BASED METHOD */
    std::cout << "\n{EVALUATION E:} -- CROSS BASED -- " << std::endl;
    cv::Mat disparity_image_cb;
    SemiGlobalMatching(left_image_clr,
                       right_image_clr,
                       disparity_image_cb,
                       STEREO_LEFT_ONLY,
                       "no_params_needed",
                       masked_depth,
                       cv::Mat(),
                       VOLUME_UPDATE_CROSS);
    if (SAVE_IMAGES) {
        std::string save_file_name = "fuse_cross_based.png";
        std::string save_url = save_dir + save_file_name;
        std::cout << "{NS} saving image to: " << save_url << std::endl;
        cv::imwrite(save_url, disparity_image_cb, compression_params);
    }
    cv::Mat_<double> disp_CB = disparity_image_cb / SCALING_FACTOR;
    /* evaluate neighborhood support method */
    cv::Mat_<double> error_image_cb;
    double average_error_cb;
    utilities.calculateAccuracy(disp_CB,
                                disp_image,
                                average_error_cb,
                                error_image_cb,
                                sample_mask);
    std::cout << "{CB} avg error: " << average_error_cb << std::endl;


    return 0;
}
