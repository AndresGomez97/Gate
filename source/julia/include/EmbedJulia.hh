#ifndef EMBED_JULIA_HH
#define EMBED_JULIA_HH

#include <stdio.h>
#include <dlfcn.h>

#include "julia.h"

// typedefs for casting methods from libjulia.so
// Julia basic embedding
typedef void (*t_jl_init)(void);
typedef jl_value_t *(*t_jl_eval_string)(const char*);
typedef int (*t_jl_atexit_hook)(int);

// Arrays
typedef jl_value_t *(*t_jl_apply_array_type)(jl_value_t*,size_t);
typedef jl_array_t *(*t_jl_ptr_to_array)(jl_value_t*, void*,size_t,int);

// Calling Julia methods
typedef jl_value_t *(*t_jl_get_global)(jl_module_t*, jl_sym_t*);
typedef jl_value_t *(*t_jl_call)(jl_function_t*, jl_value_t**,int32_t);
typedef jl_sym_t *(*t_jl_symbol)(const char*);

// Box and Unbox types
typedef jl_value_t *(*t_jl_box_uint32)(uint32_t);
typedef jl_value_t *(*t_jl_box_int32)(int32_t);
typedef jl_value_t *(*t_jl_box_int64)(int64_t);
typedef int64_t (*t_jl_unbox_int64)(jl_value_t*);
typedef jl_value_t *(*t_jl_box_float32)(float);

//push and pop GC
typedef jl_ptls_t (*t_jl_get_ptls_states)(void);

struct RetProjection{
    jl_array_t *px; 
    jl_array_t *py; 
    jl_array_t *pz;
    jl_array_t *hole;
};

// Get handler libjulia.so
void *p_handle(void);

// Julia basic embedding
void p_jl_init(void);
int p_jl_atexit_hook(int v);
jl_value_t *p_jl_eval_string(const char *v);

// Arrays
jl_value_t *p_jl_apply_array_type(jl_value_t *type,size_t dim);
jl_array_t *p_jl_ptr_to_array_1d(jl_value_t *atype, void *data, size_t nel, int ownbuffer);

// Calling Julia methods
jl_value_t *p_jl_get_global(jl_module_t *m, jl_sym_t *var);
jl_value_t *p_jl_call(jl_function_t *f, jl_value_t **args, int32_t nargs);
jl_sym_t *p_jl_symbol(const char *str);

// Box and Unbox types
jl_value_t *p_jl_box_uint32(uint32_t x);
jl_value_t *p_jl_box_int32(int32_t x);
jl_value_t *p_jl_box_int64(int64_t x);
jl_value_t *p_jl_box_float32(float x);
int64_t p_jl_unbox_int64(jl_value_t *v);

#endif