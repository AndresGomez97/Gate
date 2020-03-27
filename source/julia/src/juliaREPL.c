#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <julia.h>
#include <juliaREPL.h>

//JULIA_DEFINE_FAST_TLS()

void calljuliaRELP()
{
  void *handle;
  void (*jl_init)(void);
  int (*jl_atexit_hook)(int);
  jl_value_t *(*jl_eval_string)(const char*);

  handle = dlopen("/home/agmez/julia-1.3.1/lib/libjulia.so", RTLD_LAZY | RTLD_GLOBAL);
  if (!handle) {
      fprintf(stderr, "%s\n", dlerror());
      exit(EXIT_FAILURE);
  }

  dlerror();

  *(void**)(&jl_init) = dlsym(handle, "jl_init__threading");
  *(int**)(&jl_atexit_hook)= dlsym(handle, "jl_atexit_hook");
  *(void**)(&jl_eval_string) = dlsym(handle, "jl_eval_string");

  dlerror();

  jl_init();

  jl_eval_string("Base._start()");

  jl_atexit_hook(0);
}
