#include <stdio.h>
#include <julia.h>
#include <dlfcn.h>
#include "test.h"
//JULIA_DEFINE_FAST_TLS() // only define this once, in an executable (not in a shared library) if you$

void calljulia()
{
  
  //HANDLER FOR DLOPEN
  void *handle;
  
  //WRONG WAY FOR CASTING METHODS WITH DLSYM C++
  /*
  void (*jl_init)(void);
  int (*jl_atexit_hook)(int);
  jl_value_t *(*jl_eval_string)(const char*);
  double (*jl_unbox_float64)(jl_value_t*);
  */

  //CORRECT WAY FOR CASTING METHODS WITH DLSYM C++
  typedef void (*t_jl_init)(void);
  typedef jl_value_t *(*t_jl_eval_string)(const char*);
  typedef int (*t_jl_atexit_hook)(int);
  typedef double (*t_jl_unbox_float64)(jl_value_t*);


  handle = dlopen("/home/agmez/julia-1.3.1/lib/libjulia.so", RTLD_LAZY | RTLD_GLOBAL);
  if (!handle) {
      fprintf(stderr, "%s\n", dlerror());
      exit(EXIT_FAILURE);
  }

  dlerror();
  
  //INCORRECT CASTING METHODS FROM LIBJULIA
  /*
  *(void**)(&jl_init) = dlsym(handle, "jl_init__threading");
  *(int **)(&jl_atexit_hook)= dlsym(handle, "jl_atexit_hook");
  *(void**)(&jl_eval_string) = dlsym(handle, "jl_eval_string");
  *(double **)(&jl_unbox_float64)= dlsym(handle,"jl_unbox_float64");
  */

  //CASTING METHODS FROM LIBJULIA
  t_jl_init jl_init = (t_jl_init)dlsym(handle, "jl_init__threading");
  t_jl_atexit_hook jl_atexit_hook= (t_jl_atexit_hook)dlsym(handle, "jl_atexit_hook");
  t_jl_eval_string jl_eval_string = (t_jl_eval_string)dlsym(handle, "jl_eval_string");
  t_jl_unbox_float64 jl_unbox_float64= (t_jl_unbox_float64)dlsym(handle,"jl_unbox_float64");

  dlerror();

  jl_init();

  jl_value_t *ret = jl_eval_string("sqrt(2.0)");

  double ret_unboxed = jl_unbox_float64(ret);
  printf("sqrt(2.0) in C: %e \n", ret_unboxed);

  jl_atexit_hook(0);

  //return 0;
}
