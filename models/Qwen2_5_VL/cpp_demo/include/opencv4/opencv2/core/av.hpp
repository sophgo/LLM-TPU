#ifndef _CORE_AV_HPP
#define _CORE_AV_HPP

#include "opencv2/core/cvdef.h"

extern "C" {
#include "libavcodec/avcodec.h"
}

namespace cv { namespace av {

CV_EXPORTS MatAllocator* getAllocator();
/* function: create an opencv specific AVFrame structure.
 * parameter:
 *      height:  height of picture or video
 *      width:   width of picture or Video
 *      color_format: AV_PIX_FMT_GRAY8, AV_PIX_FMT_GBRP, AV_PIX_FMT_YUV420P, AV_PIX_FMT_NV12,
 *                    AV_PIX_FMT_YUV422P horizontal, AV_PIX_FMT_YUV444P, AV_PIX_FMT_NV16
 *      data:    system memory address. if null, it will allocate it inside
 *      addr:    device memory address.
 *      fd:      device memory handle. if negative, it means no device memory is given, and it
 *               does not to be allocated. In soc mode, it is is ion handle. In pcie mode, give 0
 *               if device memory is valid, otherwise give -1.
 *      plane_stride: step array for each plane of memory
 *      plane_size: size for each plane of memory
 *      color_space: color space for this frame. BT601, BT709 and etc.
 *      color_range: color range for this frame. mpeg, jpeg and etc.
 *      id:      pcie card number. In soc mode, it is zero.
 *  return:
 *      AVFrame pointer. If failed, return NULL.
 **/
CV_EXPORTS AVFrame *create(int height, int width, int color_format,
                           void* data, bm_int64 addr, int fd, int* plane_stride,
                           int* plane_size, int color_space = AVCOL_SPC_BT709,
                           int color_range = AVCOL_RANGE_MPEG, int id = 0);
CV_EXPORTS AVFrame *create(int height, int width, int id = 0);
CV_EXPORTS void flush(AVFrame *frame);
CV_EXPORTS int copy(AVFrame *src, AVFrame *dst, int id);
CV_EXPORTS int get_scale_and_plane(int color_format, int wscale[], int hscale[]);

}}

#endif
