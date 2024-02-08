
%module pyglm2

%{
    #define SWIG_FILE_WITH_INIT
    #include "chatglm_c.h"
%}

%include "numpy.i"

%init %{
import_array();
%}

%apply (int* IN_ARRAY1, int DIM1) { (int* input, int length)};
%apply (int* IN_ARRAY1, int DIM1) { (int* input_tokens, int input_tokens_length),
                                    (int* eos_ids, int eos_ids_num)}
%apply  (int** ARGOUTVIEW_ARRAY1, int* DIM1 ) {(int**    result_tokens, int*    result_length)}
%include "chatglm_c.h"