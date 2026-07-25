#ifndef OPENCV_IMGPROC_BMCV_HPP
#define OPENCV_IMGPROC_BMCV_HPP

#ifdef HAVE_BMCV

#include "opencv2/core/types.hpp"

#include "bmcv_api.h"
#include "bmcv_api_ext.h"

namespace cv { namespace bmcv {

CV_EXPORTS bm_handle_t getCard(int id = 0);
CV_EXPORTS int getId(bm_handle_t handle);
CV_EXPORTS bm_device_mem_t getDeviceMem(bm_uint64 addr, int len);
CV_EXPORTS bm_status_t attachDeviceMemory(Mat &m);

CV_EXPORTS bm_status_t toBMI(Mat &m, bm_image *image, bool update = true);
CV_EXPORTS bm_status_t toBMI(Mat &m, Mat &m1, Mat &m2, Mat &m3, bm_image *image, bool update = true);

CV_EXPORTS bm_status_t toMAT(Mat &in, Mat &m0, bool update = true);

CV_EXPORTS bm_status_t toMAT(bm_image *image, Mat &m0, bool update = true, csc_type_t csc = CSC_MAX_ENUM);
CV_EXPORTS bm_status_t toMAT(bm_image *image, Mat &m0, Mat &m1, Mat &m2, Mat &m3, bool update = true, csc_type_t csc = CSC_MAX_ENUM);
/* toMAT: this function is superset. It supports convertsion like toMAT(bm_image, Mat, update, csc), also supports
 * direct conversion without copy.
 * parameter:
 *      image - input bm_image, 1N mode
 *      m - output Mat
 *      color_space - [AVCOL_SPC_BT709/AVCOL_SPC_BT470], defined in FFMPEG pixfmt.h, BT601/709
 *      color_range - [AVCOL_RANGE_MPEG/AVCOL_RANGE_JPEG], defined in FFMPEG pixfmt.h, MPEG/JPEG(full-range) color range for yuv format
 *      vaddr - system memory pointer. If null, allocate it internally
 *      fd0 - physical address handle. If negative, use handle in bm_image
 *      update - If true, sync device memory to system memory. Otherwise not.
 *      nocopy - If true, direct conversion without copy. Otherwise, convert to BGR standard Mat formnvert.
 *  return:
 *      BM_SUCCESS - success
 *      BM_NOT_SUPPORTED - failed.
 */
CV_EXPORTS bm_status_t toMAT(bm_image *image, Mat &m, int color_space, int color_range, void* vaddr = NULL, int fd0 = -1,
                             bool update = true, bool nocopy = true);

CV_EXPORTS bm_status_t decomp(Mat &m, Mat &out, bool update = true);
CV_EXPORTS bm_status_t resize(Mat &m, Mat &out, bool update = true, int interpolation = BMCV_INTER_NEAREST);
CV_EXPORTS bm_status_t convert(Mat &m, Mat &out, bool update = true);

CV_EXPORTS bm_status_t convert(Mat &m, std::vector<Rect> &vrt, std::vector<Size> &vsz, std::vector<Mat> &out, \
        bool update = true, csc_type_t csc = CSC_YCbCr2RGB_BT601, csc_matrix_t * matrix = nullptr, \
        bmcv_resize_algorithm algorithm = BMCV_INTER_LINEAR);
CV_EXPORTS bm_status_t convert(Mat &m, std::vector<Rect> &vrt, bm_image *out, bool update = true);
CV_EXPORTS void uploadMat(Mat &mat);
CV_EXPORTS void downloadMat(Mat &mat);
CV_EXPORTS bm_status_t stitch(std::vector<Mat>& in, std::vector<Rect>& srt, std::vector<Rect>& drt, Mat &out, \
        bool update = true,  bmcv_resize_algorithm algorithm = BMCV_INTER_LINEAR);

CV_EXPORTS bm_status_t hwColorCvt(Mat &srcm, OutputArray _dst, int srcformat, int dstformat, Size dstSz, int dcn);
CV_EXPORTS bm_status_t hwResize(Mat &srcm, OutputArray _dst, Size dsize, int interpolation);
CV_EXPORTS bm_status_t hwBorder(Mat &srcm, int top, int bottom, int left, int right,OutputArray _dst,const Scalar& value);
CV_EXPORTS bm_status_t hwCrop(InputArray _src, Rect rect , OutputArray _dst);
CV_EXPORTS bm_status_t hwMultiCrop(InputArray _src, std::vector<Rect>& loca, std::vector<Mat>& dst_vector);
CV_EXPORTS bm_status_t hwSobel(InputArray _src, OutputArray _dst, int dx, int dy, int ksize, double scale, double delta);
CV_EXPORTS bm_status_t hwGaussianBlur(InputArray _src, OutputArray _dst, int kw, int kh, double sigma1, double sigma2);

CV_EXPORTS void print(Mat &m, bool dump = false);
CV_EXPORTS void print(bm_image *image, bool dump = false);
CV_EXPORTS void dumpMat(Mat &image, const String &fname);
CV_EXPORTS void dumpBMImage(bm_image *image, const String &fname);
}}

#endif
#endif
