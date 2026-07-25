#ifndef __CV_BMCPU_HPP__
#define __CV_BMCPU_HPP__

namespace cv{

#define OCV_BMCPU_ALIGN(x, align)   (((x) + ((align)-1)) & ~((align)-1))

#define BMCPU_LOG(console, formats, ...) \
        fprintf(console, "[%s:%d->%s]" formats, __FILE__, __LINE__, __func__, ##__VA_ARGS__)


class CV_EXPORTS BMCpuSender{
    public:
        BMCpuSender(int dev_id = 0, int size = 8192);
        ~BMCpuSender();

        int run(const String& function_name);
        int put(Mat &m);
        int skip(Mat &m);
        int get(Mat &m);
        int put(String &text);
        int skip(String &text);
        int get(String &text);
        int put(unsigned char &m);
        int skip(unsigned char &m);
        int get(unsigned char &m);
        int put(int &m);
        int skip(int &m);
        int get(int &m);
        int put(float &m);
        int skip(float &m);
        int get(float &m);
        int put(double &m);
        int skip(double &m);
        int get(double &m);
        int put(bool &m);
        int skip(bool &m);
        int get(bool &m);
        int put(void *m, int byte_size);
        int skip(void *m, int &byte_size);
        int get(void **m, int &byte_size);

        template <typename _Tp> int put(std::vector<_Tp> &vec)
        {
            int ret = 0;
            int position = _param_byte_size;

            if (!_buffer){
                BMCPU_LOG(stderr, "BMCpuSender device %d buffer is not allocated\n", _device_id);
                return -1;
            }
            if (_param_byte_size + sizeof(int64_t) > (size_t)_buffer_size){
                BMCPU_LOG(stderr, "BMCpuSender no sufficient memory for parameters\n");
                return -1;
            }

            *(int*)(_buffer+_param_byte_size) = vec.size();
            _param_byte_size += sizeof(int64_t);

            for (size_t i = 0; i < vec.size(); i++){
                if (typeid(Mat) == typeid(vec[i]))
                    ret = put((Mat &)vec[i]);
                else if (typeid(unsigned char) == typeid(vec[i]))
                    ret = put((unsigned char &)vec[i]);
                else if (typeid(int) == typeid(vec[i]))
                    ret = put((int &)vec[i]);
                else if (typeid(float) == typeid(vec[i]))
                    ret = put((float &)vec[i]);
                else if (typeid(double) == typeid(vec[i]))
                    ret = put((double &)vec[i]);
                else if (typeid(bool) == typeid(vec[i]))
                    ret = put((bool &)vec[i]);
                else if (typeid(String) == typeid(vec[i]))
                    ret = put((String &)vec[i]);
                else if (typeid(Point) == typeid(vec[i]))
                    ret = put((Point &)vec[i]);
                else if (typeid(Point2f) == typeid(vec[i]))
                    ret = put((Point2f &)vec[i]);
                else if (typeid(Scalar) == typeid(vec[i]))
                    ret = put((Scalar &)vec[i]);
                else if (typeid(Size) == typeid(vec[i]))
                    ret = put((Size &)vec[i]);
                else if (typeid(Size2f) == typeid(vec[i]))
                    ret = put((Size2f &)vec[i]);
                else {
                    BMCPU_LOG(stderr, "BMCpuSender no supported data type, report it and add\n");
                    _param_byte_size = position;
                    return -1;
                }

                if (ret < 0){
                    _param_byte_size = position;
                    return ret;
                }
            }

            return ret;
        }

        template <typename _Tp> int skip(std::vector<_Tp> &vec)
        {
            int ret = 0;
            int position = _param_byte_size;
            int num;

            if (!_buffer){
                BMCPU_LOG(stderr, "BMCpuSender device %d buffer is not allocated\n", _device_id);
                return -1;
            }
            if (_param_byte_size + sizeof(int64_t) > (size_t)_buffer_size){
                BMCPU_LOG(stderr, "BMCpuSender device %d no sufficient memory for parameters\n", _device_id);
                return -1;
            }

            num = *(int*)(_buffer+_param_byte_size);
            _param_byte_size += sizeof(int64_t);

            for (int i = 0; i < num; i++){
                if (typeid(Mat) == typeid(vec[0]))
                    ret = skip((Mat &)vec[0]);
                else if (typeid(unsigned char) == typeid(vec[0]))
                    ret = skip((unsigned char &)vec[0]);
                else if (typeid(int) == typeid(vec[0]))
                    ret = skip((int &)vec[0]);
                else if (typeid(float) == typeid(vec[i]))
                    ret = skip((float &)vec[i]);
                else if (typeid(double) == typeid(vec[0]))
                    ret = skip((double &)vec[0]);
                else if (typeid(bool) == typeid(vec[0]))
                    ret = skip((bool &)vec[0]);
                else if (typeid(String) == typeid(vec[0]))
                    ret = skip((String &)vec[0]);
                else if (typeid(Point) == typeid(vec[0]))
                    ret = skip((Point &)vec[0]);
                else if (typeid(Point2f) == typeid(vec[i]))
                    ret = skip((Point2f &)vec[i]);
                else if (typeid(Scalar) == typeid(vec[0]))
                    ret = skip((Scalar &)vec[0]);
                else if (typeid(Size) == typeid(vec[0]))
                    ret = skip((Size &)vec[0]);
                else if (typeid(Size2f) == typeid(vec[0]))
                    ret = skip((Size2f &)vec[0]);
                else {
                    BMCPU_LOG(stderr, "BMCpuSender no supported data type, report it and add\n");
                    _param_byte_size = position;
                    return -1;
                }

                if (ret < 0){
                    _param_byte_size = position;
                    return ret;
                }
            }

            return ret;
        }

        template <typename _Tp> int get(std::vector<_Tp> &vec)
        {
            int ret = 0;
            int position = _param_byte_size;
            int num;

            if (!_buffer){
                BMCPU_LOG(stderr, "BMCpuSender device %d buffer is not allocated\n", _device_id);
                return -1;
            }
            if (_param_byte_size + sizeof(int64_t) > (size_t)_buffer_size){
                BMCPU_LOG(stderr, "BMCpuSender device %d no sufficient memory for parameters\n", _device_id);
                return -1;
            }

            num = *(int*)(_buffer + _param_byte_size);
            _param_byte_size += sizeof(int64_t);

            if (vec.size() != (size_t)num && typeid(Mat) == typeid(_Tp)){
                BMCPU_LOG(stderr, "BMCpuSender error: vector size is not as same as number given in parameter buffer (Mat case), initialized it outside\n");
                _param_byte_size = position;
                return -1;
            }

            for (int i = 0; i < num; i++){
                _Tp x;
                if (vec.size() <= (size_t)i)
                    vec.push_back(x);
                if (typeid(Mat) == typeid(vec[i]))
                    ret = get((Mat&)vec[i]);
                else if (typeid(unsigned char) == typeid(vec[i]))
                    ret = get((unsigned char&)vec[i]);
                else if (typeid(int) == typeid(vec[i]))
                    ret = get((int&)vec[i]);
                else if (typeid(float) == typeid(vec[i]))
                    ret = get((float &)vec[i]);
                else if (typeid(double) == typeid(vec[i]))
                    ret = get((double&)vec[i]);
                else if (typeid(bool) == typeid(vec[i]))
                    ret = get((bool&)vec[i]);
                else if (typeid(String) == typeid(vec[i]))
                    ret = get((String&)vec[i]);
                else if (typeid(Point) == typeid(vec[i]))
                    ret = get((Point&)vec[i]);
                else if (typeid(Point2f) == typeid(vec[i]))
                    ret = get((Point2f &)vec[i]);
                else if (typeid(Scalar) == typeid(vec[i]))
                    ret = get((Scalar&)vec[i]);
                else if (typeid(Size) == typeid(vec[i]))
                    ret = get((Size&)vec[i]);
                else if (typeid(Size2f) == typeid(vec[i]))
                    ret = get((Size2f&)vec[i]);
                else {
                    BMCPU_LOG(stderr, "BMCpuSender no supported data type, report it and add\n");
                    _param_byte_size = position;
                    return -1;
                }

                if (ret < 0){
                    _param_byte_size = position;
                    return ret;
                }
            }

            return ret;
        }

        template <typename _Tp> int put(Point_<_Tp> &p)
        {
            int ret = 0;
            if (!_buffer){
                BMCPU_LOG(stderr, "BMCpuSender device %d buffer is not allocated\n", _device_id);
                return -1;
            }
            if (_param_byte_size + 2*sizeof(_Tp) > (size_t)_buffer_size){
                BMCPU_LOG(stderr, "BMCpuSender no sufficient memory for parameters\n");
                return -1;
            }

            *(_Tp *)(_buffer+_param_byte_size) = p.x;
            _param_byte_size += sizeof(_Tp);
            *(_Tp *)(_buffer+_param_byte_size) = p.y;
            _param_byte_size += sizeof(_Tp);
            _param_byte_size = OCV_BMCPU_ALIGN(_param_byte_size, sizeof(int64_t));

            return ret;
        }

        template <typename _Tp> int skip(Point_<_Tp> &p)
        {
            int ret = 0;
            if (!_buffer){
                BMCPU_LOG(stderr, "BMCpuSender device %d buffer is not allocated\n", _device_id);
                return -1;
            }
            if (_param_byte_size + 2*sizeof(_Tp) > (size_t)_buffer_size){
                BMCPU_LOG(stderr, "BMCpuSender no sufficient memory for parameters\n");
                return -1;
            }

            _param_byte_size += 2*sizeof(_Tp);
            _param_byte_size = OCV_BMCPU_ALIGN(_param_byte_size, sizeof(int64_t));

            return ret;

        }

        template <typename _Tp> int get(Point_<_Tp> &p)
        {
            int ret = 0;
            if (!_buffer){
                BMCPU_LOG(stderr, "BMCpuSender device %d buffer is not allocated\n", _device_id);
                return -1;
            }
            if (_param_byte_size + 2*sizeof(_Tp) > (size_t)_buffer_size){
                BMCPU_LOG(stderr, "BMCpuSender no sufficient memory for parameters\n");
                return -1;
            }

            p.x = *(_Tp *)(_buffer+_param_byte_size);
            _param_byte_size += sizeof(_Tp);
            p.y = *(_Tp *)(_buffer+_param_byte_size);
            _param_byte_size += sizeof(_Tp);
            _param_byte_size = OCV_BMCPU_ALIGN(_param_byte_size, sizeof(int64_t));

            return ret;
        }

        template <typename _Tp> int put(Scalar_<_Tp> &s)
        {
            int ret = 0;
            if (!_buffer){
                BMCPU_LOG(stderr, "BMCpuSender device %d buffer is not allocated\n", _device_id);
                return -1;
            }
            if (_param_byte_size + 4*sizeof(_Tp) > (size_t)_buffer_size){
                BMCPU_LOG(stderr, "BMCpuSender no sufficient memory for parameters\n");
                return -1;
            }

            for (int i = 0; i < 4; i++){
                *(_Tp *)(_buffer+_param_byte_size) = s.val[i];
                _param_byte_size += sizeof(_Tp);
            }

            _param_byte_size = OCV_BMCPU_ALIGN(_param_byte_size, sizeof(int64_t));

            return ret;

        }

        template <typename _Tp> int skip(Scalar_<_Tp> &s)
        {
            int ret = 0;
            if (!_buffer){
                BMCPU_LOG(stderr, "BMCpuSender device %d buffer is not allocated\n", _device_id);
                return -1;
            }
            if (_param_byte_size + 4*sizeof(_Tp) > (size_t)_buffer_size){
                BMCPU_LOG(stderr, "BMCpuSender no sufficient memory for parameters\n");
                return -1;
            }

            _param_byte_size += 4*sizeof(_Tp);
            _param_byte_size = OCV_BMCPU_ALIGN(_param_byte_size, sizeof(int64_t));

            return ret;

        }

        template <typename _Tp> int get(Scalar_<_Tp> &s)
        {
            int ret = 0;
            if (!_buffer){
                BMCPU_LOG(stderr, "BMCpuSender device %d buffer is not allocated\n", _device_id);
                return -1;
            }
            if (_param_byte_size + 4*sizeof(_Tp) > (size_t)_buffer_size){
                BMCPU_LOG(stderr, "BMCpuSender no sufficient memory for parameters\n");
                return -1;
            }

            for (int i = 0; i < 4; i++){
                s.val[i] = *(_Tp *)(_buffer+_param_byte_size);
                _param_byte_size += sizeof(_Tp);
            }

            _param_byte_size = OCV_BMCPU_ALIGN(_param_byte_size, sizeof(int64_t));

            return ret;

        }

        template <typename _Tp> int put(Size_<_Tp> &size)
        {
            int ret = 0;
            if (!_buffer){
                BMCPU_LOG(stderr, "BMCpuSender device %d buffer is not allocated\n", _device_id);
                return -1;
            }
            if (_param_byte_size + 2*sizeof(_Tp) > (size_t)_buffer_size){
                BMCPU_LOG(stderr, "BMCpuSender no sufficient memory for parameters\n");
                return -1;
            }

            *(_Tp *)(_buffer+_param_byte_size) = size.width;
            _param_byte_size += sizeof(_Tp);
            *(_Tp *)(_buffer+_param_byte_size) = size.height;
            _param_byte_size += sizeof(_Tp);

            _param_byte_size = OCV_BMCPU_ALIGN(_param_byte_size, sizeof(int64_t));

            return ret;

        }

        template <typename _Tp> int skip(Size_<_Tp> &size)
        {
            int ret = 0;
            if (!_buffer){
                BMCPU_LOG(stderr, "BMCpuSender device %d buffer is not allocated\n", _device_id);
                return -1;
            }
            if (_param_byte_size + 2*sizeof(_Tp) > (size_t)_buffer_size){
                BMCPU_LOG(stderr, "BMCpuSender no sufficient memory for parameters\n");
                return -1;
            }

            _param_byte_size += 2*sizeof(_Tp);
            _param_byte_size = OCV_BMCPU_ALIGN(_param_byte_size, sizeof(int64_t));

            return ret;

        }


        template <typename _Tp> int get(Size_<_Tp> &size)
        {
            int ret = 0;
            if (!_buffer){
                BMCPU_LOG(stderr, "BMCpuSender device %d buffer is not allocated\n", _device_id);
                return -1;
            }
            if (_param_byte_size + 2*sizeof(_Tp) > (size_t)_buffer_size){
                BMCPU_LOG(stderr, "BMCpuSender no sufficient memory for parameters\n");
                return -1;
            }

            size.width = *(_Tp *)(_buffer+_param_byte_size);
            _param_byte_size += sizeof(_Tp);
            size.height = *(_Tp *)(_buffer+_param_byte_size);
            _param_byte_size += sizeof(_Tp);

            _param_byte_size = OCV_BMCPU_ALIGN(_param_byte_size, sizeof(int64_t));

            return ret;

        }

    private:
        template <typename _Tp> int put_basic_data(_Tp &value)
        {
            int ret = 0;
            if (!_buffer){
                BMCPU_LOG(stdout, "BMCpuSender device %d buffer is not allocated\n", _device_id);
                return -1;
            }
            if (_param_byte_size + sizeof(_Tp) > (size_t)_buffer_size){
                BMCPU_LOG(stdout, "BMCpuSender no sufficient memory for parameters\n");
                return -1;
            }

            *(_Tp *)(_buffer+_param_byte_size) = value;
            _param_byte_size += sizeof(_Tp);
            _param_byte_size = OCV_BMCPU_ALIGN(_param_byte_size, sizeof(int64_t));

            return ret;
        }

        template <typename _Tp> int skip_basic_data(_Tp &value)
        {
            int ret = 0;

            CV_UNUSED(value);

            if (!_buffer){
                BMCPU_LOG(stdout, "BMCpuSender device %d buffer is not allocated\n", _device_id);
                return -1;
            }
            if (_param_byte_size + sizeof(_Tp) > (size_t)_buffer_size){
                BMCPU_LOG(stdout, "BMCpuSender no sufficient memory for parameters\n");
                return -1;
            }

            _param_byte_size += sizeof(_Tp);
            _param_byte_size = OCV_BMCPU_ALIGN(_param_byte_size, sizeof(int64_t));

            return ret;
        }

        template <typename _Tp> int get_basic_data(_Tp &value)
        {
            int ret = 0;
            if (!_buffer){
                BMCPU_LOG(stdout, "BMCpuSender device %d buffer is not allocated\n", _device_id);
                return -1;
            }
            if (_param_byte_size + sizeof(_Tp) > (size_t)_buffer_size){
                BMCPU_LOG(stdout, "BMCpuSender no sufficient memory for parameters\n");
                return -1;
            }

            value = *(_Tp *)(_buffer+_param_byte_size);
            _param_byte_size += sizeof(_Tp);
            _param_byte_size = OCV_BMCPU_ALIGN(_param_byte_size, sizeof(int64_t));

            return ret;
        }

        int size(Mat &m);

        unsigned char *_buffer;
        int _param_byte_size;
        int _buffer_size;
        int _device_id;
        unsigned char *_map_vaddr;
        bm_handle_t _handle;
        int _process_handle;
        bm_device_mem_t _dev_mem;
};

}

#endif
