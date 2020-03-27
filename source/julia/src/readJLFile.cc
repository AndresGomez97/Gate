#include <stdio.h>
#include <julia.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include "readJLFile.hh"

#define FILE_OK 0
#define FILE_NOT_EXIST 1
#define FILE_TO_LARGE 2
#define FILE_READ_ERROR 3

char * readFile(char * filename) {
    FILE *f = fopen(filename, "rt");
    assert(f);
    fseek(f, 0, SEEK_END);
    long length = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buffer = (char *) malloc(length + 1);
    buffer[length] = '\0';
    fread(buffer, 1, length, f);
    fclose(f);
    return buffer;
}


//JULIA_DEFINE_FAST_TLS() // only define this once, in an executable (not in a shared library) if you$

void readJLFile(char * filename)
{
  
  //HANDLER FOR DLOPEN
  void *handle;

  //CORRECT WAY FOR CASTING METHODS WITH DLSYM C++
  typedef void (*t_jl_init)(void);
  typedef jl_value_t *(*t_jl_eval_string)(const char*);
  typedef int (*t_jl_atexit_hook)(int);


  handle = dlopen("/home/agmez/julia-1.3.1/lib/libjulia.so", RTLD_LAZY | RTLD_GLOBAL);
  if (!handle) {
      fprintf(stderr, "%s\n", dlerror());
      exit(EXIT_FAILURE);
  }

  dlerror();

  //CASTING METHODS FROM LIBJULIA
  t_jl_init jl_init = (t_jl_init)dlsym(handle, "jl_init__threading");
  t_jl_atexit_hook jl_atexit_hook= (t_jl_atexit_hook)dlsym(handle, "jl_atexit_hook");
  t_jl_eval_string jl_eval_string = (t_jl_eval_string)dlsym(handle, "jl_eval_string");

  dlerror();

  char *res= readFile(filename);
  
  jl_init();

  jl_eval_string(res);

  jl_atexit_hook(0);

}
