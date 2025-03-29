#ifndef __BM_CORE_HAL_HPP
#define __BM_CORE_HAL_HPP

#ifndef WIN32
#include "opencv2/core/ion.hpp"
#endif

namespace cv { namespace hal {

CV_EXPORTS MatAllocator* getAllocator();

}}

#endif
