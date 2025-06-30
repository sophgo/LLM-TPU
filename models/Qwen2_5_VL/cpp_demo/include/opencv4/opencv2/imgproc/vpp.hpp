#ifndef OPENCV_VPP_HPP
#define OPENCV_VPP_HPP

#include "opencv2/core/types_c.h"

typedef  char CSC_TYPE;

namespace cv { namespace vpp {
CV_EXPORTS void enableVPPConverter( bool isEnabled );
CV_EXPORTS bool IsVPPConverterEnabled();
CV_EXPORTS bool vppCvtColor(Mat& src, Mat&dst, int srcFmt, int dstFmt);

/** @brief Resizes an image by vpp hardware accelerator.

The function is hardware acceleartor version of cv::resize(Mat, Mat).
you may call the function as follows:
@code
    Mat src = imread("pic.jpg");
    Mat dst(h, w, CV_8UC3);
    vpp::resize(src, dst);
@endcode

@param src input image, can be Mat
@param dst output image; it has the size dsize (when it is non-zero) or the size computed from
src.size(), fx, and fy; the type of dst is as same as of src 
*/
CV_EXPORTS void resize(Mat& src, Mat& dst);             //C++

/** @brief Python API: Resizes an image by vpp hardware accelerator.

The function is python API fo hardware acceleartor version of cv::resize(Mat, Mat).
In python, you can call the function in this way:
@code
    import cv2 as cv
    src = cv.imread("pic.jpg")
    # here dst is UMat object, dst.get() will return a numpy array.
    dst = cv.vpp.resize(src,dst)
@endcode

@param _src input image, can be Mat or UMat. If _src is Mat, an extra copy will be happened in function 
      due to numpy memory allocation.
@param _dst output image; it has the size dsize (when it is non-zero) or the size computed from
src.size(), fx, and fy; the type of dst is UMat. If python need numpy array, dst.get() will return numpy array.
*/

CV_EXPORTS_W void resize(InputArray _src, CV_OUT UMat& _dst);           //python

/** @brief draw border around source image by hardware accelerator
The function is hardware acceleartor version cv::copyMakeBorder to draw border around source image. 
The result of new image stores in a new Mat. This result Mat is created inside function. The 
output new image size is (left+src.width+right, top+src.height+bottom)

you may call the function as follows:
@code
    Mat src(height, width, CV_8UC3); 
    Mat dst = vpp::border(src, top, bottom, left, right);
@endcode

@param src input image.
@param top the pixel width of top border
@param bottom the pixel width of bottom border
@param left the pixel width of left border
@param right the pixel width of righ border

@return new Mat object contain source image content and borders.
*/
CV_EXPORTS Mat border(Mat& src, int top, int bottom, int left, int right);      // C++

/** @brief Python API: draw border around source image by hardware accelerator
The function is hardware acceleartor version cv::copyMakeBorder to draw border around source image. 
The result of new image stores in a new UMat. This result UMat is created inside function. The 
output new image size is (left+src.width+right, top+src.height+bottom). In python, returned dst is
UMat object, dst.get() will copy UMat data to numpy array.

If you want to use it python, you can call the function this way:
@code
    import cv2 as cv
    src = cv.imread("pic.jpg")
    dst = cv.vpp.border(src, top, bottom, left, right)
@endcode

@param src input image, which can be in Mat or UMat format. When Mat(numpy array) uses, addtional 
       memcpy will be happened in function to ION memory.
@param top the pixel width of top border
@param bottom the pixel width of bottom border
@param left the pixel width of left border
@param right the pixel width of righ border

@return new UMat object contain source image content and borders.
*/

CV_EXPORTS_W void border(InputArray src, int top, int bottom, int left, int right, CV_OUT UMat& dst);  // python

/** @brief convert RGB pack format of source image to RGB planar format by hardware accelerator
The function seperates r/g/b channel into output image 
array.
you may call the function as follows:
@code
    Mat dst[3];
    Mat src = imread("pic.jpg");
    vpp::split(src, dst);
@endcode

@param src input image in MAT type.
@param dst array point of output image channel in MAT, 0 - R, 1 -G, 2 - B

@return void
*/
CV_EXPORTS void split(Mat& src, Mat* dst);        // C++

/** @brief Python API: convert RGB pack format of source image to RGB planar format by hardware accelerator
The function seperates r/g/b channel into output image 
array.
you may call the function as follows:
@code
    import cv2 as cv
    src = cv.imread("pic.jpg")
    dst = cv.vpp.split(src, dst = []);
    for i in range(3):
      cv.imwrite("{:d}.jpg".format(i), dst[i])
@endcode

@param src input image in MAT or UMAT format. Mat need an extra memcpy inside function.
@param dst array point of output image channel in UMAT format, 0 - R, 1 -G, 2 - B. UMat.get() can get numpy
       dat by memcpy

@return void
*/
CV_EXPORTS_W void split(InputArray src, CV_OUT std::vector<UMat>& dst);   // python


/** @brief convert RGB pack format of source image to RGB planar format by hardware accelerator.
The function seperates r/g/b channel into three specific
address. This funciton does not have python implementation, because it has not pointer data type.
you may call the function as follows:
@code
    Mat src(height, width, CV_8UC3); 
    for (int i = 0; i < 3; i++)
      dst[i] = Mat(height, width, CV_8UC1);    
    vpp::split(src, dst[0].data, dst[1].data, dst[2].data);
@endcode

@param src input image.
@param addr0 output adress R of Vpp ION memory
@param addr1 output adress G of Vpp ION memory
@param addr2 output adress B of Vpp ION memory

@return void
*/
CV_EXPORTS void split(Mat& src, uint64_t addr0, uint64_t addr1, uint64_t addr2);  //C++

/** @brief convert RGB pack format to RGB planar format by hardware accelerator
The function seperates r/g/b channel into Mat data type.
you may call the function as follows:
@code
    Mat src(height, width, CV_8UC3); 
    Mat dst(height, width, CV_8UC3); 
    ...
    vpp::split(src, dst);     // convert from pack to planar format
@endcode

@param src input image in Mat type.
@param dst output image in Mat type. It has the size dsize (when it is non-zero) or the size computed from
src.size(). The type of dst is UMat. If python need numpy array, dst.get() will return numpy array.

@return void
*/
CV_EXPORTS void split(Mat& src, Mat& dst);        // C++

/** @brief Python API: convert RGB pack format to RGB planar format by hardware accelerator
The function seperates r/g/b channel into Mat data type.
you may call the function as follows:
@code
    import cv2 as cv
    src = cv.imread("pic.jpg")
    dst = cv.UMat(src.shape[0], src.shape[1], cv.CV_8UC3)
    dst = cv.vpp.split(src, dst)
    dst.get().tofile("out.bin")
    cv.imwrite("out.jpg", dst)
@endcode

@param src input image in MAT or UMAT format. Mat need an extra memcpy inside function.
@param dst output image in UMAT format. it has the size dsize (when it is non-zero) or the size computed from
src.size(). The type of dst is UMat. If python need numpy array, dst.get() will return numpy array.

@return void
*/
CV_EXPORTS_W void split(InputArray src, CV_OUT UMat& dst);    // python

/** @brief crop source image into several small images by hardware accelerator
The function crops a image into sevreal small images depending on given rectangel 
information. The output small image is allocated inside function.
you may call the function as follows:
@code
    Mat src(height, width, CV_8UC3); // or use Mat to replace createVppMat
    vector<Rect> loca;
    ...
    Mat dst[N];
    vpp::crop(src, loca, dst);
    ...
@endcode

@param src one input image, Mat type.
@param local cropping vector of rectangle coordinations
@param dst array pointer of multiple output image, pointer of Mat.

@return void
*/
CV_EXPORTS void crop(Mat& src, std::vector<Rect>& loca, Mat* dst);      // C++

/** @brief Python API: crop source image into several small images by hardware accelerator
The function crops a image into sevreal small images depending on given rectangel information. 
The output small image is allocated inside function.
you may call the function as follows:

If you want to use it python, you can call the function this way:
@code
    import cv2 as cv
    import numpy as np    
    src = cv.imread("pic.jpg")
    w = 1920
    h = 1080
    N = 16
    cw = int(w/4)
    ch = int(h/4)
    loca = [(i*cw, j*ch, cw, ch) for i in range(4) for j in range(4)]
    dst = cv.vpp.crop(src, local, dst)
    for i in range(N):
      cv.imwrite("{:d}.bmp".format(i), dst[i])
@endcode

@param src one input image, this can be Mat, MatVector, UMat or UMatVector. If this is Mat or MatVector, 
       an extra copy is happened in function because python use numpy memory allocation for Mat. 
@param local cropping vector of rectangle coordinations
@param dst vector of multiple output image, vector of UMat. In python, cv.UMat.get() method can
       get numpy array or Mat type. cv.UMat.get() is another memcpy to numpy memory.

@return void
*/
CV_EXPORTS_W void crop(InputArray _src, std::vector<Rect>& _loca, CV_OUT std::vector<UMat>& _dst);    // python
CV_EXPORTS_W void crop(InputArrayOfArrays _src, std::vector<Rect>& _loca, CV_OUT std::vector<UMat>& _dst);    // python


/** @brief crop source image to output image one by one by hardware accelerator
The function output every image cropping from each source image by every rectangle coordination. 
The output images are allocated inside function. 
you may call the function as follows:
@code
    Mat src(w,h,type)[N];
    vector<Rect> loca;
    ...
    Mat dst[N];
    vpp::crop(src, loca, dst);
    ...
@endcode

@param src multiple input image.
@param local cropping vector of rectangle coordinations
@param dst array pointer of multiple output image

@return void
*/
CV_EXPORTS void crop(Mat* src, std::vector<Rect>& loca, Mat* dst);      // C++


/** @brief This is experimental function. It will crop rectangle from source images, and paste
it to different localtion of destination image, i.e. mosaic effect. The output images are allocated 
inside function. 
you may call the function as follows:
@code
    Mat src(w,h,type)[N];
    vector<Rect> loca;
    ...
    Mat dst[N];
    vpp::crop(src, loca, dst);
    ...
@endcode

@param src multiple input image.
@param local cropping vector of rectangle coordinations
@param dst array pointer of multiple output image

@return void
*/

CV_EXPORTS void crop(Mat* src, std::vector<Rect>& loca, Mat& dst);


/** @brief conver NV12 to BGR24 pack format
This function convert video image data to BGR24 image format. IplImage data has NV12 format,
which comes from video decoding, picture decoding and etc. The output is BGR24 pack format, 
which is common inside opencv. 
This function does not has python interface, and generally used as private usage.
The output Mat data is allocated inside function.

@param img image data in NV12 format

@return Mat output data in BGR24 format
*/
CV_EXPORTS Mat iplImageToMat(IplImage* img);
/*fix by lh*/

CV_EXPORTS Mat iplImageToMat(IplImage* img, int test_flg);

/** @brief conver NV12 to BGR24 pack format, the conversion formula is given by csc_type.
This function convert video image data to BGR24 image format. IplImage data has NV12 format,
which comes from video decoding, picture decoding and etc. The output is BGR24 pack format, 
which is common inside opencv. The color conversion formula is decided by csc_type.
This function does not has python interface, and generally used as private usage.
The output Mat data is allocated inside function.

@param img image data in NV12 format
@param csc_type 0 - default(YCbCr to RGB) 1 - YPbPr to RGB

@return Mat output data in BGR24 format
*/
CV_EXPORTS Mat iplImageToMat(IplImage* img, CSC_TYPE csc_type);

/* add by jinf */
/** @brief convert endian of RGB24, such as BGR24 to RGB24.
This function convert RGB endian, such as BGR24 to RGB24, or other endians.
This function does not has python interface, and generally used as private usage.
The output Mat data is allocated inside function.
Now, only BGR24toRGB24 is verified. 

@param src image data
@param srcFmt source format
@param dstFmt output format

@return Mat output data
*/
CV_EXPORTS Mat cvtColor1682(Mat& src, int srcFmt, int dstFmt);

/** @brief convert endian of RGB24, such as BGR24 to RGB24.
This function convert RGB endian, such as BGR24 to RGB24, or other endians.
This function does not has python interface, and generally used as private usage.
The output image data is allocated out of function, and given in IplImage data type.
Now, only BGR24toRGB24 is verified. 

@param src image data
@param img output image data
@param srcFmt source format
@param dstFmt output format

@return Mat output data
*/
CV_EXPORTS void cvtColor1682(Mat& src, IplImage* img, int srcFmt, int dstFmt);

/** @brief convert Mat with AVFrame to Mat with BGR24

@param img input image data
@param out output image data

*/
CV_EXPORTS void toMat(Mat& img, Mat& out);

CV_EXPORTS void setDump(int dumpNum);
}}

/*color space BM1682 supported*/
#define FMT_SRC_I420    (0)
#define FMT_SRC_NV12    (1)
#define FMT_SRC_RGBP    (2)
#define FMT_SRC_BGRA    (3)
#define FMT_SRC_RGBA    (4)
#define FMT_SRC_BGR     (5)
#define FMT_SRC_RGB     (6)

#define FMT_DST_I420    (0)
#define FMT_DST_Y       (1)
#define FMT_DST_RGBP    (2)
#define FMT_DST_BGRA    (3)
#define FMT_DST_RGBA    (4)
#define FMT_DST_BGR     (5)
#define FMT_DST_RGB     (6)

#ifdef VPP_BM1880
/*color space BM1880 supported*/
#define YUV420        0
#define YOnly         1
#define RGB24         2
#define ARGB32        3

/*maximum and minimum image resolution BM1880 supported*/
#define MAX_RESOLUTION_W    (1920)
#define MAX_RESOLUTION_H    (1440)

#define MIN_RESOLUTION_W_LINEAR    (8)    /*linear mode to linear mode*/
#define MIN_RESOLUTION_H_LINEAR    (8)    /*linear mode to linear mode*/
#define MIN_RESOLUTION_W_TILE1    (10)    /*nv12 tile to yuv420p*/
#define MIN_RESOLUTION_H_TILE1    (10)    /*nv12 tile to yuv420p*/
#define MIN_RESOLUTION_W_TILE2    (9)    /*nv12 tile to rgb24, rgb32, rgbp*/
#define MIN_RESOLUTION_H_TILE2    (9)    /*nv12 tile to rgb24, rgb32, rgbp*/
#define MIN_RESOLUTION_W_TILE3    (8)    /*yonly tile to yonly linear*/
#define MIN_RESOLUTION_H_TILE3    (8)    /*yonly tile to yonly linear*/
#else
/*maximum and minimum image resolution BM1682 supported*/
#define MAX_RESOLUTION_W    (4096)
#define MAX_RESOLUTION_H    (4096)
#ifdef MIN_RESOLUTION_W
#undef MIN_RESOLUTION_W
#endif
#define MIN_RESOLUTION_W    (16)
#ifdef MIN_RESOLUTION_H
#undef MIN_RESOLUTION_H
#endif
#define MIN_RESOLUTION_H    (16)
#endif

#ifdef VPP_BM1684
/*color space BM1880 supported*/
#define YUV420        0
#define YOnly         1
#define RGB24         2
#define ARGB32        3

/*maximum and minimum image resolution BM1880 supported*/
#define MAX_RESOLUTION_W    (4096)
#define MAX_RESOLUTION_H    (4096)

#define MIN_RESOLUTION_W_LINEAR    (8)    /*linear mode to linear mode*/
#define MIN_RESOLUTION_H_LINEAR    (8)    /*linear mode to linear mode*/

struct cv_csc_matrix {
	int csc_coe00;
	int csc_coe01;
	int csc_coe02;
	int csc_add0;
	int csc_coe10;
	int csc_coe11;
	int csc_coe12;
	int csc_add1;
	int csc_coe20;
	int csc_coe21;
	int csc_coe22;
	int csc_add2;
};
#endif

#define ION_CACHE

enum csc_coe_type {
  Default = 0, YPbPr
};

struct vpp_cmd {
  int src_format;
  int src_stride;
#if (defined VPP_BM1880) || (defined VPP_BM1684)
  int src_endian;
  int src_endian_a;
  int src_plannar;
#endif
  int src_fd0;
  int src_fd1;
  int src_fd2;
  unsigned long src_addr0;
  unsigned long src_addr1;
  unsigned long src_addr2;
  unsigned short src_axisX;
  unsigned short src_axisY;
  unsigned short src_cropW;
  unsigned short src_cropH;

  int dst_format;
  int dst_stride;
#if (defined VPP_BM1880) || (defined VPP_BM1684)
  int dst_endian;
  int dst_endian_a;
  int dst_plannar;
#endif
  int dst_fd0;
  int dst_fd1;
  int dst_fd2;
  unsigned long dst_addr0;
  unsigned long dst_addr1;
  unsigned long dst_addr2;
  unsigned short dst_axisX;
  unsigned short dst_axisY;
  unsigned short dst_cropW;
  unsigned short dst_cropH;
#if defined VPP_BM1880
  int tile_mode;
#elif defined VPP_BM1684
  int src_csc_en;
#endif

#if (defined VPP_BM1880) || (defined VPP_BM1684)
  int hor_filter_sel;
  int ver_filter_sel;
  int scale_x_init;
  int scale_y_init;
#endif
  int csc_type;
#if (defined VPP_BM1880) || (defined VPP_BM1684)
  int mapcon_enable;
  int src_fd3;
  unsigned long src_addr3;
  int cols;
  int rows;
#endif
#if defined VPP_BM1684
  int src_uv_stride;
  int dst_uv_stride;
  struct cv_csc_matrix matrix;
#endif
};

#define VPP_MAX_BATCH_NUM  16
struct vpp_batch {
  int num;
  struct vpp_cmd cmd[VPP_MAX_BATCH_NUM];
};

#define VPP_UPDATE_BATCH _IOWR('v', 0x01, unsigned long)
#define VPP_UPDATE_BATCH_VIDEO _IOWR('v', 0x02, unsigned long)
#define VPP_UPDATE_BATCH_SPLIT _IOWR('v', 0x03, unsigned long)
#define VPP_UPDATE_BATCH_NON_CACHE _IOWR('v', 0x04, unsigned long)
#define VPP_UPDATE_BATCH_CROP_TEST _IOWR('v', 0x05, unsigned long)
#define VPP_UPDATE_BATCH_FD_PA _IOWR('v', 0x09, unsigned long)

#endif
