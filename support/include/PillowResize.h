/*
* Copyright 2021 Zuru Tech HK Limited
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* istributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#ifndef PILLOWRESIZE_HPP
#define PILLOWRESIZE_HPP

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cstdlib>
#include <cmath>
#include <cstring>

enum PixelType {
    UINT8,
    INT8,
    UINT16,
    INT16,
    INT32,
    FLOAT32
};

typedef struct ResizeMat_s {
    unsigned short  width;
    unsigned short  height;
    unsigned short  channels;
    unsigned long   data;
    PixelType       pixel_type;
} ResizeMat;

/**
 * \brief PillowResize Porting of the resize methods from Pillow library
 * (https://github.com/python-pillow/Pillow).
 * The implementation depends only on OpenCV, so all Pillow code has been
 * converted to use cv::Mat and OpenCV structures.
 * Since Pillow does not support natively all cv::Mat types, this implementation
 * extends the support to almost all OpenCV pixel types.
 */
class PillowResize {
protected:
    /**
     * \brief precision_bits 8 bits for result. Filter can have negative areas.
     * In one case the sum of the coefficients will be negative,
     * in the other it will be more than 1.0. That is why we need
     * two extra bits for overflow and int type.
     */
    static constexpr uint32_t precision_bits = 32 - 8 - 2;

    static constexpr double box_filter_support = 0.5;
    static constexpr double bilinear_filter_support = 1.;
    static constexpr double hamming_filter_support = 1.;
    static constexpr double bicubic_filter_support = 2.;
    static constexpr double lanczos_filter_support = 3.;

    /**
     * \brief Filter Abstract class to handle the filters used by
     * the different interpolation methods.
     */
    class Filter {
    private:
        double _support; /** Support size (length of resampling filter). */

    public:
        /**
         * \brief Construct a new Filter object.
         *
         * \param[in] support Support size (length of resampling filter).
         */
        explicit Filter(double support) : _support{support} {};

        /**
         * \brief filter Apply filter.
         *
         * \param[in] x Input value.
         *
         * \return Processed value by the filter.
         */
        [[nodiscard]] virtual double filter(double x) const = 0;

        /**
         * \brief support Get support size.
         *
         * \return support size.
         */
        [[nodiscard]] double support() const { return _support; };
    };

    class BoxFilter : public Filter {
    public:
        BoxFilter() : Filter(box_filter_support){};
        [[nodiscard]] double filter(double x) const override {
            constexpr double half_pixel = 0.5;
            if (x > -half_pixel && x <= half_pixel) {
                return 1.0;
            }
            return 0.0;
        }
    };

    class BilinearFilter : public Filter {
    public:
        BilinearFilter() : Filter(bilinear_filter_support){};
        [[nodiscard]] double filter(double x) const override {
            if (x < 0.0) {
                x = -x;
            }
            if (x < 1.0) {
                return 1.0 - x;
            }
            return 0.0;
        }
    };

    class HammingFilter : public Filter {
    public:
        HammingFilter() : Filter(hamming_filter_support){};
        [[nodiscard]] double filter(double x) const override {
            if (x < 0.0) {
                x = -x;
            }
            if (x == 0.0) {
                return 1.0;
            }
            if (x >= 1.0) {
                return 0.0;
            }
            x = x * M_PI;
            return sin(x) / x * (0.54 + 0.46 * cos(x));    // NOLINT
        }
    };

    class BicubicFilter : public Filter {
    public:
        BicubicFilter() : Filter(bicubic_filter_support){};
        [[nodiscard]] double filter(double x) const override {
            // https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
            constexpr double a = -0.5;
            if (x < 0.0) {
                x = -x;
            }
            if (x < 1.0) {                                         // NOLINT
                return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;    // NOLINT
            }
            if (x < 2.0) {                                 // NOLINT
                return (((x - 5) * x + 8) * x - 4) * a;    // NOLINT
            }
            return 0.0;
        }
    };

    class LanczosFilter : public Filter {
    protected:
        [[nodiscard]] static double _sincFilter(double x) {
            if (x == 0.0) {
                return 1.0;
            }
            x = x * M_PI;
            return sin(x) / x;
        }

    public:
        LanczosFilter() : Filter(lanczos_filter_support){};
        [[nodiscard]] double filter(double x) const override {
            // Truncated sinc.
            // According to Jim Blinn, the Lanczos kernel (with a = 3)
            // "keeps low frequencies and rejects high frequencies better
            // than any (achievable) filter we've seen so far."[3]
            // (https://en.wikipedia.org/wiki/Lanczos_resampling#Advantages)
            constexpr double lanczos_a_param = 3.0;
            if (-lanczos_a_param <= x && x < lanczos_a_param) {
                return _sincFilter(x) * _sincFilter(x / lanczos_a_param);
            }
            return 0.0;
        }
    };

#if __cplusplus >= 201703L
    /**
     * \brief _lut Generate lookup table.
     * \reference https://joelfilho.com/blog/2020/compile_time_lookup_tables_in_cpp/
     *
     * \tparam Length Number of table elements.
     * \param[in] f Functor called to generate each elements in the table.
     *
     * \return An array of length Length with type deduced from Generator output.
     */
    template <size_t Length, typename Generator>
    static constexpr auto _lut(Generator&& f)
    {
        using content_type = decltype(f(size_t{0}));
        std::array<content_type, Length> arr{};
        for (size_t i = 0; i < Length; ++i) {
            arr[i] = f(i);
        }
        return arr;
    }

    /**
     * \brief _clip8_lut Clip lookup table.
     *
     * \tparam Length Number of table elements.
     * \tparam min_val Value of the starting element.
     */
    template <size_t Length, intmax_t min_val>
    static inline constexpr auto _clip8_lut =
        _lut<Length>([](size_t n) -> uint8_t {
            intmax_t saturate_val = static_cast<intmax_t>(n) + min_val;
            if (saturate_val < 0) {
                return 0;
            }
            if (saturate_val > UINT8_MAX) {
                return UINT8_MAX;
            }
            return static_cast<uint8_t>(saturate_val);
        });
#endif

    /**
     * \brief _clip8 Optimized clip function.
     *
     * \param[in] in input value.
     *
     * \return Clipped value.
     */
    [[nodiscard]] static uint8_t _clip8(double in)
    {
#if __cplusplus >= 201703L
        // Lookup table to speed up clip method.
        // Handles values from -640 to 639.
        const uint8_t* clip8_lookups =
            &_clip8_lut<1280, -640>[640];    // NOLINT
        // NOLINTNEXTLINE
        return clip8_lookups[static_cast<intmax_t>(in) >> precision_bits];
#else
        auto saturate_val = static_cast<intmax_t>(in) >> precision_bits;
        if (saturate_val < 0) {
            return 0;
        }
        if (saturate_val > UINT8_MAX) {
            return UINT8_MAX;
        }
        return static_cast<uint8_t>(saturate_val);
#endif
    }

    /**
     * \brief _roundUp Round function.
     * The output value will be cast to type T.
     *
     * \param[in] f Input value.
     *
     * \return Rounded value.
     */
    template <typename T>
    [[nodiscard]] static T _roundUp(double f)
    {
        return static_cast<T>(std::round(f));
    }


    /**
     * \brief _precomputeCoeffs Compute 1D interpolation coefficients.
     * If you have an image (or a 2D matrix), call the method twice to compute
     * the coefficients for row and column either.
     * The coefficients are computed for each element in range [0, out_size).
     *
     * \param[in] in_size Input size (e.g. image width or height).
     * \param[in] in0 Input starting index.
     * \param[in] in1 Input last index.
     * \param[in] out_size Output size.
     * \param[in] filterp Pointer to a Filter object.
     * \param[out] bounds Bounds vector. A bound is a pair of xmin and xmax.
     * \param[out] kk Coefficients vector. To each elements corresponds a number of
     * coefficients returned by the function.
     *
     * \return Size of the filter coefficients.
     */
    [[nodiscard]] static int32_t _precomputeCoeffs(
        int32_t in_size,
        double in0,
        double in1,
        int32_t out_size,
        const std::shared_ptr<Filter>& filterp,
        std::vector<int32_t>& bounds,
        std::vector<double>& kk)
    {
        // Prepare for horizontal stretch.
        const double scale = (in1 - in0) / static_cast<double>(out_size);
        double filterscale = scale;
        if (filterscale < 1.0) {
            filterscale = 1.0;
        }

        // Determine support size (length of resampling filter).
        const double support = filterp->support() * filterscale;

        // Maximum number of coeffs.
        const auto k_size = static_cast<int32_t>(ceil(support)) * 2 + 1;

        // Check for overflow
        if (out_size >
            INT32_MAX / (k_size * static_cast<int32_t>(sizeof(double)))) {
            throw std::runtime_error("Memory error");
        }

        // Coefficient buffer.
        kk.resize(out_size * k_size);

        // Bounds vector.
        bounds.resize(out_size * 2);

        int32_t x = 0;
        constexpr double half_pixel = 0.5;
        for (int32_t xx = 0; xx < out_size; ++xx) {
            double center = in0 + (xx + half_pixel) * scale;
            double ww = 0.0;
            double ss = 1.0 / filterscale;
            // Round the value.
            auto xmin = static_cast<int32_t>(center - support + half_pixel);
            if (xmin < 0) {
                xmin = 0;
            }
            // Round the value.
            auto xmax = static_cast<int32_t>(center + support + half_pixel);
            if (xmax > in_size) {
                xmax = in_size;
            }
            xmax -= xmin;
            double* k = &kk[xx * k_size];
            for (x = 0; x < xmax; ++x) {
                double w = filterp->filter((x + xmin - center + half_pixel) * ss);
                k[x] = w;    // NOLINT
                ww += w;
            }
            for (x = 0; x < xmax; ++x) {
                if (ww != 0.0) {
                    k[x] /= ww;    // NOLINT
                }
            }
            // Remaining values should stay empty if they are used despite of xmax.
            for (; x < k_size; ++x) {
                k[x] = 0;    // NOLINT
            }
            bounds[xx * 2 + 0] = xmin;
            bounds[xx * 2 + 1] = xmax;
        }
        return k_size;
    }

    /**
     * \brief _normalizeCoeffs8bpc Normalize coefficients for 8 bit per pixel matrix.
     *
     * \param[in] prekk Filter coefficients.
     *
     * \return Filter coefficients normalized.
     */
    [[nodiscard]] static std::vector<double> _normalizeCoeffs8bpc(
        const std::vector<double>& prekk)
    {
        std::vector<double> kk;
        kk.reserve(prekk.size());

        constexpr auto shifted_coeff = static_cast<double>(1U << precision_bits);

        constexpr double half_pixel = 0.5;
        for (const auto& k : prekk) {
            if (k < 0) {
                kk.emplace_back(trunc(-half_pixel + k * shifted_coeff));
            }
            else {
                kk.emplace_back(trunc(half_pixel + k * shifted_coeff));
            }
        }
        return kk;
    }

    using preprocessCoefficientsFn =
        std::vector<double> (*)(const std::vector<double>&);

    template <typename T>
    using outMapFn = T (*)(double);

    /**
     * \brief _resampleHorizontal Apply resample along the horizontal axis.
     *
     * \param[in, out] im_out Output resized matrix.
     *                       The matrix has to be previously initialized with right size.
     * \param[in] im_in Input matrix.
     * \param[in] offset Vertical offset (first used row in the source image).
     * \param[in] ksize Interpolation filter size.
     * \param[in] bounds Interpolation filter bounds (index of min and max pixel
     *                   to be considered by the filter).
     * \param[in] prekk Interpolation filter coefficients.
     * \param[in] preprocessCoefficients Function used to process the filter coefficients.
     * \param[in] init_buffer Initial value of pixel buffer (default: 0.0).
     * \param[in] outMap Function used to convert the value of the pixel after
     *                   the interpolation into the output pixel.
     */
    template <typename T>
    static void _resampleHorizontalImpl(
        ResizeMat im_out,
        const ResizeMat im_in,
        int32_t offset,
        int32_t ksize,
        const std::vector<int32_t>& bounds,
        const std::vector<double>& prekk,
        preprocessCoefficientsFn preprocessCoefficients = nullptr,
        double init_buffer = 0.,
        outMapFn<T> outMap = nullptr)
    {
        std::vector<double> kk(prekk.begin(), prekk.end());
        // Preprocess coefficients if needed.
        if (preprocessCoefficients != nullptr) {
            kk = preprocessCoefficients(kk);
        }

        for (int32_t yy = 0; yy < im_out.height; ++yy) {
            for (int32_t xx = 0; xx < im_out.width; ++xx) {
                const int32_t xmin = bounds[xx * 2 + 0];
                const int32_t xmax = bounds[xx * 2 + 1];
                const double* k = &kk[xx * ksize];
                for (int32_t c = 0; c < 3; ++c) {
                    double ss = 0.0;
                    for (int32_t x = 0; x < xmax; ++x) {
                        ss += static_cast<double>(
                            ((unsigned char*)im_in.data + ((yy + offset) * im_in.width + x + xmin) * im_in.channels)[c]) *
                              k[x];
                    }
                    // NOLINTNEXTLINE
                    ((unsigned char*)im_out.data + (yy * im_out.width + xx) * im_out.channels)[c] =
                        (outMap == nullptr ? static_cast<T>(ss) : outMap(ss));
                }
            }
        }
    }

    /**
     * \brief _resampleHorizontal Apply resample along the horizontal axis.
     * It calls the _resampleHorizontal with the correct pixel type using
     * the value returned by cv::Mat::type().
     *
     * \param[in, out] im_out Output resized matrix.
     *                        The matrix has to be previously initialized with right size.
     * \param[in] im_in Input matrix.
     * \param[in] offset Vertical offset (first used row in the source image).
     * \param[in] ksize Interpolation filter size.
     * \param[in] bounds Interpolation filter bounds (value of the min and max column
     *                   to be considered by the filter).
     * \param[in] prekk Interpolation filter coefficients.
     */
    static void _resampleHorizontal(ResizeMat im_out,
                                    const ResizeMat im_in,
                                    int32_t offset,
                                    int32_t ksize,
                                    const std::vector<int32_t>& bounds,
                                    const std::vector<double>& prekk)
    {
        switch (im_in.pixel_type) {
            case UINT8:
                return _resampleHorizontalImpl<uint8_t>(
                    im_out, im_in, offset, ksize, bounds, prekk,
                    _normalizeCoeffs8bpc, static_cast<double>(1U << (precision_bits - 1U)), _clip8);
            default:
                throw std::runtime_error("Unsupported pixel type");
        }
    }

    /**
     * \brief _resample Resize a matrix using the specified interpolation method.
     *
     * \param[in] im_in Input matrix.
     * \param[in] im_out Output matrix.
     * \param[in] x_size Desidered output width.
     * \param[in] y_size Desidered output height.
     * \param[in] filter_p Pointer to the interpolation filter.
     * \param[in] rect Input region that has to be resized.
     *                 Region is defined as a vector of 4 point x0,y0,x1,y1.
     */
    static void _resample(
        const ResizeMat im_in,
        const ResizeMat im_out,
        int32_t x_size,
        int32_t y_size,
        const std::shared_ptr<Filter>& filter_p,
        const float rect[4])
    {
        ResizeMat im_temp;

        std::vector<int32_t> bounds_horiz;
        std::vector<int32_t> bounds_vert;
        std::vector<double> kk_horiz;
        std::vector<double> kk_vert;

        const bool need_horizontal = x_size != im_in.width ||
                                     (rect[0] != 0.0F) ||
                                     static_cast<int32_t>(rect[2]) != x_size;
        const bool need_vertical = y_size != im_in.height ||
                                   (rect[1] != 0.0F) ||
                                   static_cast<int32_t>(rect[3]) != y_size;

        // Compute horizontal filter coefficients.
        const int32_t ksize_horiz =
            _precomputeCoeffs(im_in.width, rect[0], rect[2], x_size,
                              filter_p, bounds_horiz, kk_horiz);

        // Compute vertical filter coefficients.
        const int32_t ksize_vert =
            _precomputeCoeffs(im_in.height, rect[1], rect[3], y_size,
                              filter_p, bounds_vert, kk_vert);

        // First used row in the source image.
        const int32_t ybox_first = bounds_vert[0];
        // Last used row in the source image.
        const int32_t ybox_last =
            bounds_vert[y_size * 2 - 2] + bounds_vert[y_size * 2 - 1];

        // Two-pass resize, horizontal pass.
        if (need_horizontal) {
            // Shift bounds for vertical pass.
            for (int32_t i = 0; i < y_size; ++i) {
                bounds_vert[i * 2] -= ybox_first;
            }

            // Create destination image with desired ouput width and same input pixel type.
            if (need_vertical) {
                im_temp.width = x_size;
                im_temp.height = ybox_last - ybox_first;
                im_temp.channels = im_in.channels;
                im_temp.pixel_type = im_in.pixel_type;
                im_temp.data = (unsigned long)malloc(im_temp.width * im_temp.height * im_in.channels);
            } else
                im_temp = im_out;
            _resampleHorizontal(im_temp, im_in, ybox_first, ksize_horiz,
                                bounds_horiz, kk_horiz);
        }

        // Vertical pass.
        if (need_vertical) {
            // Create destination image with desired ouput size and same input pixel type.
            if (!need_horizontal)
                im_temp = im_in;
            // Input can be the original image or horizontally resampled one.
            _resampleVertical(im_out, im_temp, ksize_vert, bounds_vert,
                                kk_vert);
            if (need_horizontal)
                free((void*)im_temp.data);
        }
        return;
    }

    /**
     * \brief _nearestResample Resize a matrix using nearest neighbor interpolation.
     *
     * \param[in] im_in Input matrix.
     * \param[in] im_out Output matrix.
     * \param[in] x_size Desidered output width.
     * \param[in] y_size Desidered output height.
     * \param[in] rect Input region that has to be resized.
     *                 Region is defined as a vector of 4 point x0,y0,x1,y1.
     */
    static void _nearestResample(const ResizeMat im_in,
                                        const ResizeMat im_out,
                                        int32_t x_size,
                                        int32_t y_size,
                                        const float rect[4])
    {
        auto rx0 = static_cast<int32_t>(rect[0]);
        auto ry0 = static_cast<int32_t>(rect[1]);
        auto rx1 = static_cast<int32_t>(rect[2]);
        auto ry1 = static_cast<int32_t>(rect[3]);
        rx0 = std::max(rx0, 0);
        ry0 = std::max(ry0, 0);
        rx1 = std::min(rx1, (int32_t)im_in.width);
        ry1 = std::min(ry1, (int32_t)im_in.height);

        double m[2][3] =
            {{static_cast<double>(rx1 - rx0) / static_cast<double>(x_size), 0, static_cast<double>(rx0)},
            {0, static_cast<double>(ry1 - ry0) / static_cast<double>(y_size), static_cast<double>(ry0)}};

        // Check pixel type and determine the pixel size
        // (element size * number of channels).
        size_t pixel_size = 0;
        switch (im_in.pixel_type) {
            case UINT8:
                pixel_size = sizeof(uint8_t);
                break;
            case INT8:
                pixel_size = sizeof(int8_t);
                break;
            case UINT16:
                pixel_size = sizeof(uint16_t);
                break;
            case INT16:
                pixel_size = sizeof(int16_t);
                break;
            case INT32:
                pixel_size = sizeof(int32_t);
                break;
            case FLOAT32:
                pixel_size = sizeof(float);
                break;
            default:
                throw std::runtime_error("Pixel type not supported");
        }
        pixel_size *= im_in.channels;

        const int32_t x0 = 0;
        const int32_t y0 = 0;
        const int32_t x1 = x_size;
        const int32_t y1 = y_size;

        double xo = m[0][2] + m[0][0] * 0.5;
        double yo = m[1][2] + m[1][1] * 0.5;

        auto coord = [](double x) -> int32_t {
            return x < 0. ? -1 : static_cast<int32_t>(x);
        };

        std::vector<int> xintab;
        xintab.resize(im_out.width);

        /* Pretabulate horizontal pixel positions */
        int32_t xmin = x1;
        int32_t xmax = x0;
        for (int32_t x = x0; x < x1; ++x) {
            int32_t xin = coord(xo);
            if (xin >= 0 && xin < im_in.width) {
                xmax = x + 1;
                if (x < xmin) {
                    xmin = x;
                }
                xintab[x] = xin;
            }
            xo += m[0][0];
        }

        for (int32_t y = y0; y < y1; ++y) {
            int32_t yi = coord(yo);
            if (yi >= 0 && yi < im_in.height) {
                for (int32_t x = xmin; x < xmax; ++x) {
                    memcpy((unsigned char*)(im_out.data) + (y * im_out.width + x) * pixel_size,
                        (unsigned char*)(im_in.data) + (yi * im_in.width + xintab[x]) * pixel_size, pixel_size);
                }
            }
            yo += m[1][1];
        }

        return;
    }

    /**
     * \brief transpose Transpose the image matrix.
     *
     * \param[in, out] dst Destination matrix.
     * \param[in] src Source matrix.
     */
    static void transpose(ResizeMat dst,
                   const ResizeMat src) {
        size_t h = src.height;
        size_t w = src.width;
        size_t c = src.channels;
        unsigned char * dst_addr = (unsigned char *)dst.data;
        unsigned char * src_addr = (unsigned char *)src.data;

        for (size_t i = 0; i < h; ++i) {
            for (size_t j = 0; j < w; ++j) {
                dst_addr[(j * h + i) * c] = src_addr[(i * w + j) * c];
                dst_addr[(j * h + i) * c + 1] = src_addr[(i * w + j) * c + 1];
                dst_addr[(j * h + i) * c + 2] = src_addr[(i * w + j) * c + 2];
            }
        }
    }

    /**
     * \brief _resampleVertical Apply resample along the vertical axis.
     * It calls the _resampleVertical with the correct pixel type using
     * the value returned by cv::Mat::type().
     *
     * \param[in, out] im_out Output resized matrix.
     *                        The matrix has to be previously initialized with right size.
     * \param[in] im_in Input matrix.
     * \param[in] ksize Interpolation filter size.
     * \param[in] bounds Interpolation filter bounds (value of the min and max row
     *                   to be considered by the filter).
     * \param[in] prekk Interpolation filter coefficients.
     */
    static void _resampleVertical(ResizeMat im_out,
                                  const ResizeMat im_in,
                                  int32_t ksize,
                                  const std::vector<int32_t>& bounds,
                                  const std::vector<double>& prekk)
    {
        ResizeMat im_int, im_outt;
        im_int.width = im_in.height;
        im_int.height = im_in.width;
        im_int.channels = im_in.channels;
        im_int.pixel_type = im_in.pixel_type;
        im_int.data = (unsigned long)malloc(im_int.width * im_int.height * im_int.channels);
        im_outt.width = im_out.height;
        im_outt.height = im_out.width;
        im_outt.channels = im_out.channels;
        im_outt.pixel_type = im_out.pixel_type;
        im_outt.data = (unsigned long)malloc(im_outt.width * im_outt.height * im_int.channels);

        transpose(im_int, im_in);
        _resampleHorizontal(im_outt, im_int, 0, ksize, bounds, prekk);

        transpose(im_out, im_outt);
        free((void*)im_int.data);
        free((void*)im_outt.data);
        return;
    }

public:
    /**
     * \brief InterpolationMethods Interpolation methods.
     *
     * \see https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-filters.
     */
    enum InterpolationMethods {
        INTERPOLATION_NEAREST = 0,
        INTERPOLATION_BOX = 4,
        INTERPOLATION_BILINEAR = 2,
        INTERPOLATION_HAMMING = 5,
        INTERPOLATION_BICUBIC = 3,
        INTERPOLATION_LANCZOS = 1,
    };

    /**
     * \brief resize Porting of Pillow resize method.
     *
     * \param[in] src Input matrix that has to be processed.
     * \param[in] out Output matrix that will store the resized image.
     * \param[in] filter Interpolation method code, see InterpolationMethods.
     * \param[in] box Input roi. Only the elements inside the box will be resized.
     *
     * \return None. The resized image is stored in 'out'.
     *
     * \throw std::runtime_error In case the box is invalid, the interpolation filter
     *        or the input matrix type are not supported.
     */
    static void resize(const ResizeMat src,
                                        const ResizeMat out,
                                        int32_t filter,
                                        const float box[4])
    {
        // Box = x0,y0,w,h
        // Rect = x0,y0,x1,y1
        const float rect[4] = {0, 0, box[2], box[3]};

        const int32_t x_size = out.width;
        const int32_t y_size = out.height;
        const int32_t c_size = out.channels;
        if (x_size < 1 || y_size < 1) {
            throw std::runtime_error("Height and width must be > 0");
        }

        if (rect[0] < 0.F || rect[1] < 0.F) {
            throw std::runtime_error("Box offset can't be negative");
        }

        if (static_cast<int32_t>(rect[2]) > src.width ||
            static_cast<int32_t>(rect[3]) > src.height) {
            throw std::runtime_error("Box can't exceed original image size");
        }

        if (box[2] < 0 || box[3] < 0) {
            throw std::runtime_error("Box can't be empty");
        }

        // If box's coordinates are int and box size matches requested size
        if (static_cast<int32_t>(box[2]) == x_size &&
            static_cast<int32_t>(box[3]) == y_size) {
            memcpy((void*)out.data, (void*)src.data, x_size * y_size * c_size);
            return;
        }
        if (filter == INTERPOLATION_NEAREST) {
            _nearestResample(src, out, x_size, y_size, rect);
            return;
        }
        std::shared_ptr<Filter> filter_p;

        // Check filter.
        switch (filter) {
            case INTERPOLATION_BOX:
                filter_p = std::make_shared<BoxFilter>(BoxFilter());
                break;
            case INTERPOLATION_BILINEAR:
                filter_p = std::make_shared<BilinearFilter>(BilinearFilter());
                break;
            case INTERPOLATION_HAMMING:
                filter_p = std::make_shared<HammingFilter>(HammingFilter());
                break;
            case INTERPOLATION_BICUBIC:
                filter_p = std::make_shared<BicubicFilter>(BicubicFilter());
                break;
            case INTERPOLATION_LANCZOS:
                filter_p = std::make_shared<LanczosFilter>(LanczosFilter());
                break;
            default:
                throw std::runtime_error("unsupported resampling filter");
        }

        _resample(src, out, x_size, y_size, filter_p, rect);
        return;
    }

    /**
     * \brief resize Porting of Pillow resize method.
     *
     * \param[in] src Input matrix that has to be processed.
     * \param[in] out Output matrix that will store the resized image.
     * \param[in] filter Interpolation method code, see InterpolationMethods.
     *
     * \return None. The resized image is stored in 'out'.
     *
     * \throw std::runtime_error In case the interpolation filter
     *        or the input matrix type are not supported.
     */
    static void resize(const ResizeMat src,
                                        const ResizeMat out,
                                        int32_t filter)
    {
        float box[4] = {0.F, 0.F, static_cast<float>(src.width), static_cast<float>(src.height)};
        resize(src, out, filter, box);
        return;
    }
};

#endif // PILLOWRESIZE_HPP
