#include <stdio.h>
#include <dlfcn.h>

#include "julia.h"
#include "EmbedJulia.hh"

void *p_handle(void){
    void *handle = dlopen("/home/agmez/julia-1.3.1/lib/libjulia.so", RTLD_LAZY | RTLD_GLOBAL); 
    if(!handle){
        fprintf(stderr, "%s\n", dlerror());
        exit(EXIT_FAILURE);
    }
    return handle;
}

void p_jl_init(void){
    t_jl_init jl_init = (t_jl_init)dlsym(p_handle(), "jl_init__threading");
    jl_init();
}
int p_jl_atexit_hook(int v){
    t_jl_atexit_hook jl_atexit_hook = (t_jl_atexit_hook)dlsym(p_handle(), "jl_atexit_hook");
    return jl_atexit_hook(v);
}
jl_value_t *p_jl_eval_string(const char *v){
    t_jl_eval_string jl_eval_string = (t_jl_eval_string)dlsym(p_handle(), "jl_eval_string");
    return jl_eval_string(v);
}
jl_value_t *p_jl_apply_array_type(jl_value_t *type,size_t dim){
    t_jl_apply_array_type jl_apply_array_type = (t_jl_apply_array_type)dlsym(p_handle(),"jl_apply_array_type");
    return jl_apply_array_type(type,dim);
}
jl_array_t *p_jl_ptr_to_array_1d(jl_value_t *atype, void *data, size_t nel, int ownbuffer){
    t_jl_ptr_to_array jl_ptr_to_array_1d = (t_jl_ptr_to_array)dlsym(p_handle(),"jl_ptr_to_array_1d");
    return jl_ptr_to_array_1d(atype,data,nel,ownbuffer);
}

jl_array_t *p_jl_alloc_array_1d(jl_value_t *atype, size_t nr){
    t_jl_alloc_array_1d jl_alloc_array_1d = (t_jl_alloc_array_1d)dlsym(p_handle(),"jl_alloc_array_1d");
    return jl_alloc_array_1d(atype,nr);
}

jl_value_t *p_jl_get_global(jl_module_t *m, jl_sym_t *var){
    t_jl_get_global jl_get_global = (t_jl_get_global)dlsym(p_handle(),"jl_get_global");
    return jl_get_global(m,var);
}
jl_value_t *p_jl_call(jl_function_t *f, jl_value_t **args, int32_t nargs){
    t_jl_call jl_call = (t_jl_call)dlsym(p_handle(),"jl_call");
    return jl_call(f,args,nargs);
}
jl_sym_t *p_jl_symbol(const char *str){
    t_jl_symbol jl_symbol = (t_jl_symbol)dlsym(p_handle(),"jl_symbol");
    return jl_symbol(str);
}
jl_value_t *p_jl_box_uint32(uint32_t x){
    t_jl_box_uint32 jl_box_uint32 = (t_jl_box_uint32)dlsym(p_handle(),"jl_box_uint32");
    return jl_box_uint32(x);
}
jl_value_t *p_jl_box_int32(int32_t x){
    t_jl_box_int32 jl_box_int32 = (t_jl_box_int32)dlsym(p_handle(),"jl_box_int32");
    return jl_box_int32(x);
}
jl_value_t *p_jl_box_int64(int64_t x){
    t_jl_box_int64 jl_box_int64 = (t_jl_box_int64)dlsym(p_handle(),"jl_box_int64");
    return jl_box_int64(x);
}
jl_value_t *p_jl_box_float32(float x){
    t_jl_box_float32 jl_box_float32 = (t_jl_box_float32)dlsym(p_handle(),"jl_box_float32");
    return jl_box_float32(x);
}
int64_t p_jl_unbox_int64(jl_value_t *v){
    t_jl_unbox_int64 jl_unbox_int64 = (t_jl_unbox_int64)dlsym(p_handle(),"jl_unbox_int64");
    return jl_unbox_int64(v);
}