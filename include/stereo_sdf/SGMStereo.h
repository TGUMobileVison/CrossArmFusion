#pragma once
#include <png++/png.hpp>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/photo/photo.hpp>


typedef int8_t			sint8;		// �з���8λ����
typedef uint8_t			uint8;		// �޷���8λ����
typedef int16_t			sint16;		// �з���16λ����
typedef uint16_t		uint16;		// �޷���16λ����
typedef int32_t			sint32;		// �з���32λ����
typedef uint32_t		uint32;		// �޷���32λ����
typedef int64_t			sint64;		// �з���64λ����
typedef uint64_t		uint64;		// �޷���64λ����
typedef float			float32;	// �����ȸ���
typedef double			float64;	// ˫���ȸ���

class SGMStereo {
public:
  SGMStereo();

  void setDisparityTotal(const int disparityTotal);
  void setDisparityFactor(const double disparityFactor);
  void setDataCostParameters(const int sobelCapValue,
                 const int censusWindowRadius,
                 const double censusWeightFactor,
                 const int aggregationWindowRadius);
  void setSmoothnessCostParameters(const int smoothnessPenaltySmall, const int smoothnessPenaltyLarge);
  void setConsistencyThreshold(const int consistencyThreshold);

  void meshgrid(const cv::Mat &xgv, const cv::Mat &ygv, cv::Mat1i &X, cv::Mat1i &Y);
  void meshgridInit(const cv::Range &xgv, const cv::Range &ygv, cv::Mat1i &X, cv::Mat1i &Y);

  /* CROSSS BASED Method (CB) */
  void updateCostVolumes_CB(unsigned short *leftCostVolume,
                            unsigned short * rightCostVolume,
                            cv::Mat leftImage,
                            cv::Mat weightImage);

  /* Neighborhood Support Method (NS) */
  void updateCostVolumes_NS(unsigned short *leftCostVolume,
                            unsigned short * rightCostVolume,
                            cv::Mat leftImage,
                            cv::Mat weightImage);

  /* Naive Range Fusion (NRF) */
  void updateCostVolumes_NRF(unsigned short *leftCostVolume,
                             unsigned short * rightCostVolume,
                             cv::Mat leftImage,
                             cv::Mat weightImage);

  /* Diffusion Based Fusion (DB) */
  void updateCostVolumes_DB(unsigned short *leftCostVolume,
                            unsigned short * rightCostVolume,
                            cv::Mat leftImage,
                            cv::Mat weightImage);

  /* Modified original stereo compute function */
  void compute(const png::image<png::rgb_pixel>& leftImage,
               const png::image<png::rgb_pixel>& rightImage,
               float* disparityImage,
               int STEREO_MODE,
               std::string paramFile,
               cv::Mat depthImage,
               int FUSE_FLAG,
               cv::Mat leftImageFuse,
               cv::Mat weightImage);

//��ȡˮƽ��
 // void FindHorizontalArm(const sint32& x, const sint32& y, uint8& rmin, uint8& rmax)const;
// //��ȡ��ֱ��
// void SGMStereo::FindVerticalArm(const sint32& x, const sint32& y, uint8& left, uint8& right) const;


private:
  void initialize(const png::image<png::rgb_pixel>& leftImage, const png::image<png::rgb_pixel>& rightImage);
  void setImageSize(const png::image<png::rgb_pixel>& leftImage, const png::image<png::rgb_pixel>& rightImage);
  void allocateDataBuffer();
  void freeDataBuffer();
  void computeCostImage(const png::image<png::rgb_pixel>& leftImage, const png::image<png::rgb_pixel>& rightImage);
  void convertToGrayscale(const png::image<png::rgb_pixel>& leftImage,
              const png::image<png::rgb_pixel>& rightImage,
              unsigned char* leftGrayscaleImage,
              unsigned char* rightGrayscaleImage) const;
  void computeLeftCostImage(const unsigned char* leftGrayscaleImage, const unsigned char* rightGrayscaleImage,const png::image<png::rgb_pixel>& leftImage, const png::image<png::rgb_pixel>& rightImage);
  void computeCappedSobelImage(const unsigned char* image, const bool horizontalFlip, unsigned char* sobelImage) const;
  void computeCensusImage(const unsigned char* image, int* censusImage) const;
  void calcTopRowCost(unsigned char*& leftSobelRow, int*& leftCensusRow,
            unsigned char*& rightSobelRow, int*& rightCensusRow,
            unsigned short* costImageRow);
  void calcRowCosts(unsigned char*& leftSobelRow, int*& leftCensusRow,
            unsigned char*& rightSobelRow, int*& rightCensusRow,
            unsigned short* costImageRow);
  void calcPixelwiseSAD(const unsigned char* leftSobelRow, const unsigned char* rightSobelRow);
  void calcHalfPixelRight(const unsigned char* rightSobelRow);
  void addPixelwiseHamming(const int* leftCensusRow, const int* rightCensusRow);
  void computeRightCostImage();
  void performSGM(unsigned short* costImage, unsigned short* disparityImage);
  void speckleFilter(const int maxSpeckleSize, const int maxDifference, unsigned short* image) const;
  void enforceLeftRightConsistency(unsigned short* leftDisparityImage, unsigned short* rightDisparityImage) const;

  
  // Parameter
  int disparityTotal_;
  double disparityFactor_;
  int sobelCapValue_;
  int censusWindowRadius_;
  double censusWeightFactor_;
  int aggregationWindowRadius_;
  int smoothnessPenaltySmall_;
  int smoothnessPenaltyLarge_;
  int consistencyThreshold_;

  // Data
  int width_;
  int height_;
  int widthStep_;
  unsigned short* leftCostImage_;
  unsigned short* rightCostImage_;
  unsigned char* pixelwiseCostRow_;
  unsigned short* rowAggregatedCost_;
  unsigned char* halfPixelRightMin_;
  unsigned char* halfPixelRightMax_;
  int pathRowBufferTotal_;
  int disparitySize_;
  int pathTotal_;
  int pathDisparitySize_;
  int costSumBufferRowSize_;
  int costSumBufferSize_;
  int pathMinCostBufferSize_;
  int pathCostBufferSize_;
  int totalBufferSize_;
  short* sgmBuffer_;

  /* Range measurement to be fused with stereo */
  cv::Mat_<double> rangeMeasurement_;
};
