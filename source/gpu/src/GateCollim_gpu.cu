#include "GateGPUParticle.hh"
#include "GateToGPUImageSPECT.hh"

#include <dlfcn.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>

#include "julia.h"

pthread_t tid; 
pthread_mutex_t lock;

struct ThreadArg{
    GateGPUCollimator *collimator;
    GateGPUParticle *particle;
    int *hole;
};

struct RetProjection{
    jl_array_t *px; 
    jl_array_t *py; 
    jl_array_t *pz;
    jl_array_t *hole;
};

__device__ float vector_dot(float3 u, float3 v) {
    return u.x*v.x + u.y*v.y + u.z*v.z;
}

__device__ float3 vector_sub(float3 u, float3 v) {
    return make_float3(u.x-v.x, u.y-v.y, u.z-v.z);
}

__device__ float3 vector_add(float3 u, float3 v) {
    return make_float3(u.x+v.x, u.y+v.y, u.z+v.z);
}

__device__ float3 vector_mag(float3 u, float a) {
    return make_float3(u.x*a, u.y*a, u.z*a);
}

__device__ unsigned int binary_search(float position, float *tab, unsigned int maxid ) {

    unsigned short int begIdx = 0;
    unsigned short int endIdx = maxid - 1;
    unsigned short int medIdx = endIdx / 2;

    while (endIdx-begIdx > 1) {
        if (position < tab[medIdx]) {begIdx = medIdx;}
        else {endIdx = medIdx;}
        medIdx = (begIdx+endIdx) / 2;
    }
    return medIdx;
}

extern "C" {
    __global__ void kernel_map_entry(float *d_px, float *d_py, float *d_pz, 
                                    float *d_entry_collim_y, float *d_entry_collim_z,
                                    int *d_hole, unsigned int y_size, unsigned int z_size,
                                    int particle_size) {
        
        unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (id >= particle_size) {return;}
        if( d_py[ id ] > d_entry_collim_y[ 0 ] || d_py[ id ] < d_entry_collim_y[ y_size - 1 ] )
        {
            d_hole[ id ]=-1;
            return;
        }
            if( d_pz[ id ] > d_entry_collim_z[ 0 ] || d_pz[ id ] < d_entry_collim_z[ z_size - 1 ] )
        {
            d_hole[ id ] = -1;
            return;
        }

        unsigned int index_entry_y = binary_search( d_py[ id ], d_entry_collim_y, y_size );
        unsigned int index_entry_z = binary_search( d_pz[ id ], d_entry_collim_z, z_size );

        unsigned char is_in_hole_y = ( index_entry_y & 1 ) ? 0 : 1;
        unsigned char is_in_hole_z = ( index_entry_z & 1 ) ? 0 : 1;

        unsigned char in_hole = is_in_hole_y & is_in_hole_z;

        d_hole[ id ] = ( in_hole )? index_entry_y * z_size + index_entry_z : -1;
    }
}
extern "C" {
    __global__ void kernel_map_projection(float *d_px, float *d_py, float *d_pz,
                                      float *d_dx, float *d_dy, float *d_dz,
                                      int *d_hole, float planeToProject, 
                                      unsigned int particle_size) {
    
        unsigned int id = __umul24( blockIdx.x, blockDim.x ) + threadIdx.x;
        if( id >= particle_size ) return;
        if( d_hole[ id ] == -1 ) return;

        float3 n  = make_float3( -1.0f, 0.0f, 0.0f );
        float3 v0 = make_float3( planeToProject, 0.0f, 0.0f );
        float3 d  = make_float3( d_dx[ id ], d_dy[ id ], d_dz[ id ] );
        float3 p  = make_float3( d_px[ id ], d_py[ id ], d_pz[ id ] );

        float s = __fdividef( vector_dot( n, vector_sub( v0, p ) ), vector_dot( n, d ) );
        float3 newp = vector_add( p, vector_mag( d, s ) );

        d_px[id] = newp.x;
        d_py[id] = newp.y;
        d_pz[id] = newp.z;
    }
}

extern "C" {
    __global__ void kernel_map_exit(float *d_px, float *d_py, float *d_pz,
                                    float *d_exit_collim_y, float *d_exit_collim_z,
                                    int *d_hole, unsigned int y_size, unsigned int z_size,
                                    int particle_size) {
        
        unsigned int id = __umul24( blockIdx.x, blockDim.x ) + threadIdx.x;
        if( id >= particle_size ) return;
        if( d_hole[ id ] == -1 ) return;

        if( d_py[ id ] > d_exit_collim_y[ 0 ] || d_py[ id ] < d_exit_collim_y[ y_size - 1 ] )
        {
            d_hole[ id ]=-1;
            return;
        }
        if( d_pz[ id ] > d_exit_collim_z[ 0 ] || d_pz[ id ] < d_exit_collim_z[ z_size - 1 ] )
        {
            d_hole[ id ] = -1;
            return;
        }

        unsigned int index_exit_y = binary_search( d_py[ id ], d_exit_collim_y, y_size );
        unsigned int index_exit_z = binary_search( d_pz[ id ], d_exit_collim_z, z_size );

        unsigned char is_in_hole_y = ( index_exit_y & 1 )? 0 : 1;
        unsigned char is_in_hole_z = ( index_exit_z & 1 )? 0 : 1;

        unsigned char in_hole = is_in_hole_y & is_in_hole_z;

        int newhole = ( in_hole )? index_exit_y * z_size + index_exit_z : -1;

        if( newhole == -1 )
        {
            d_hole[ id ] = -1;
            return;
        }

        if( newhole != d_hole[ id ] )
        {
            d_hole[ id ] = -1;
        }
    }
}

void GateGPUCollimator_init(GateGPUCollimator *collimator) {

    cudaSetDevice(collimator->cudaDeviceID);

    unsigned int y_size = collimator->y_size;
    unsigned int z_size = collimator->z_size;

    unsigned int mem_float_y = y_size * sizeof(float);
    unsigned int mem_float_z = z_size * sizeof(float);
    

    float* d_entry_collim_y;
    float* d_entry_collim_z;
    float* d_exit_collim_y;
    float* d_exit_collim_z;
    
    cudaMalloc((void**) &d_entry_collim_y, mem_float_y);
    cudaMalloc((void**) &d_entry_collim_z, mem_float_z);
    cudaMalloc((void**) &d_exit_collim_y, mem_float_y);
    cudaMalloc((void**) &d_exit_collim_z, mem_float_z);

    cudaMemcpy(d_entry_collim_y, collimator->entry_collim_y, mem_float_y, cudaMemcpyHostToDevice);
    cudaMemcpy(d_entry_collim_z, collimator->entry_collim_z, mem_float_z, cudaMemcpyHostToDevice);
    cudaMemcpy(d_exit_collim_y, collimator->exit_collim_y, mem_float_y, cudaMemcpyHostToDevice);
    cudaMemcpy(d_exit_collim_z, collimator->exit_collim_z, mem_float_z, cudaMemcpyHostToDevice);

    collimator->gpu_entry_collim_y = d_entry_collim_y;
    collimator->gpu_entry_collim_z = d_entry_collim_z;
    collimator->gpu_exit_collim_y = d_exit_collim_y;
    collimator->gpu_exit_collim_z = d_exit_collim_z;
}

void GateGPUCollimator_process(GateGPUCollimator *collimator, GateGPUParticle *particle) {

    cudaSetDevice(collimator->cudaDeviceID);

    // Read collimator geometry
    float* d_entry_collim_y = collimator->gpu_entry_collim_y;
    float* d_entry_collim_z = collimator->gpu_entry_collim_z;
    float* d_exit_collim_y  = collimator->gpu_exit_collim_y;
    float* d_exit_collim_z  = collimator->gpu_exit_collim_z; 
    unsigned int y_size     = collimator->y_size;
    unsigned int z_size     = collimator->z_size;
    float planeToProject    = collimator->planeToProject + particle->px[0];

    // Particles allocation to the Device
    int particle_size = particle-> size;
    unsigned int mem_float_particle = particle_size * sizeof(float);
    unsigned int mem_int_hole = particle_size * sizeof(int);

    float *d_px, *d_py, *d_pz;
    float *d_dx, *d_dy, *d_dz;
    int *d_hole;
    cudaMalloc((void**) &d_px, mem_float_particle);
    cudaMalloc((void**) &d_py, mem_float_particle);
    cudaMalloc((void**) &d_pz, mem_float_particle);
    cudaMalloc((void**) &d_dx, mem_float_particle);
    cudaMalloc((void**) &d_dy, mem_float_particle);
    cudaMalloc((void**) &d_dz, mem_float_particle);
    cudaMalloc((void**) &d_hole, mem_int_hole);

    // Array of holes :)
    int *h_hole = (int*)malloc(mem_int_hole);

    // Copy particles from host to device
    cudaMemcpy(d_px, particle->px, mem_float_particle, cudaMemcpyHostToDevice);
    cudaMemcpy(d_py, particle->py, mem_float_particle, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pz, particle->pz, mem_float_particle, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dx, particle->dx, mem_float_particle, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dy, particle->dy, mem_float_particle, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dz, particle->dz, mem_float_particle, cudaMemcpyHostToDevice);

    // Kernel vars
    dim3 threads, grid;
    int block_size = 512;
    int grid_size = (particle_size + block_size - 1) / block_size;
    threads.x = block_size;
    grid.x = grid_size;

    // Kernel map entry
    kernel_map_entry<<<grid, threads>>>(d_px, d_py, d_pz, 
                                        d_entry_collim_y, d_entry_collim_z,
                                        d_hole, y_size, z_size,
                                        particle_size);

    // Kernel projection
    kernel_map_projection<<<grid, threads>>>(d_px, d_py, d_pz,
                                             d_dx, d_dy, d_dz,
                                             d_hole, planeToProject, particle_size);

    // Kernel map_exit
    kernel_map_exit<<<grid, threads>>>(d_px, d_py, d_pz,
                                       d_exit_collim_y, d_exit_collim_z,
                                       d_hole, y_size, z_size,
                                       particle_size);
    
    // Copy particles from device to host
    cudaMemcpy(particle->px, d_px, mem_float_particle, cudaMemcpyDeviceToHost);
    cudaMemcpy(particle->py, d_py, mem_float_particle, cudaMemcpyDeviceToHost);
    cudaMemcpy(particle->pz, d_pz, mem_float_particle, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_hole, d_hole, mem_int_hole, cudaMemcpyDeviceToHost);

    // Pack data to CPU
    int c = 0;
    int i = 0;
    while( i < particle_size )
    {
        if( h_hole[ i ] == -1 )
        {
            ++i;
            continue;
        }
        //h_hole[ c ] = h_hole[ i ];
        particle->px[ c ] = particle->px[ i ];
        particle->py[ c ] = particle->py[ i ];
        particle->pz[ c ] = particle->pz[ i ];
        particle->dx[ c ] = particle->dx[ i ];
        particle->dy[ c ] = particle->dy[ i ];
        particle->dz[ c ] = particle->dz[ i ];
				particle->eventID[ c ] = particle->eventID[ i ];
				particle->parentID[ c ] = particle->parentID[ i ];
				particle->trackID[ c ] = particle->trackID[ i ];
				particle->t[ c ] = particle->t[ i ];
				particle->E[ c ] = particle->E[ i ];
				particle->type[ c ] = particle->type[ i ];
        ++c;
        ++i;
    }
    particle->size = c;    

    // Free memory
    cudaFree(d_px);
    cudaFree(d_py);
    cudaFree(d_pz);
    cudaFree(d_dx);
    cudaFree(d_dy);
    cudaFree(d_dz);
    cudaFree(d_hole);

    free(h_hole);
}

void GateJuliaCollimator_process(GateGPUCollimator *collimator, GateGPUParticle *particle) {

    // Particles allocation to the Device
    int particle_size = particle-> size;
    
    // Array of holes :)
    int *hole = (int*)malloc(particle_size * sizeof(int));

    // Kernel vars
    int block_size = 512;
    int grid_size = (particle_size + block_size - 1) / block_size;

    // Read collimator geometry
    unsigned int y_size     = collimator->y_size;
    unsigned int z_size     = collimator->z_size;
    float planeToProject    = collimator->planeToProject + particle->px[0];

    // HANDLER FOR DLOPEN
    void *handle;

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
    typedef jl_value_t *(*t_jl_call0)(jl_function_t*);
    typedef jl_sym_t *(*t_jl_symbol)(const char*);

    // Box and Unbox types
    typedef jl_value_t *(*t_jl_box_uint32)(uint32_t);
    typedef jl_value_t *(*t_jl_box_int32)(int32_t);
    typedef jl_value_t *(*t_jl_box_int64)(int64_t);
    typedef int64_t (*t_jl_unbox_int64)(jl_value_t*);
    typedef jl_value_t *(*t_jl_box_float32)(float);

    //push and pop GC
    typedef jl_ptls_t (*t_jl_get_ptls_states)(void);

    // LIBJULIA
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

    t_jl_apply_array_type jl_apply_array_type = (t_jl_apply_array_type)dlsym(handle,"jl_apply_array_type");
    t_jl_ptr_to_array jl_ptr_to_array_1d = (t_jl_ptr_to_array)dlsym(handle,"jl_ptr_to_array_1d");

    t_jl_get_global jl_get_global = (t_jl_get_global)dlsym(handle,"jl_get_global");
    t_jl_call jl_call = (t_jl_call)dlsym(handle,"jl_call");
    t_jl_call0 jl_call0 = (t_jl_call0)dlsym(handle,"jl_call0");
    t_jl_symbol jl_symbol = (t_jl_symbol)dlsym(handle,"jl_symbol");

    t_jl_box_uint32 jl_box_uint32 = (t_jl_box_uint32)dlsym(handle,"jl_box_uint32");
    t_jl_box_int32 jl_box_int32 = (t_jl_box_int32)dlsym(handle,"jl_box_int32");
    t_jl_box_int64 jl_box_int64 = (t_jl_box_int64)dlsym(handle,"jl_box_int64");
    t_jl_box_float32 jl_box_float32 = (t_jl_box_float32)dlsym(handle,"jl_box_float32");
    t_jl_unbox_int64 jl_unbox_int64 = (t_jl_unbox_int64)dlsym(handle,"jl_unbox_int64");

    t_jl_get_ptls_states jl_get_ptls_states = (t_jl_get_ptls_states)dlsym(handle,"jl_get_ptls_states");

    
    jl_init();
    
    jl_datatype_t *jl_float32_type = *(jl_datatype_t **)dlsym(handle, "jl_float32_type");
    jl_datatype_t *jl_int32_type = *(jl_datatype_t **)dlsym(handle,"jl_int32_type");
    
    // Include GateKernels.jl code
    jl_eval_string("include(\"/home/agmez/gate/Gate/source/julia/jl/src/JuliaKernels.jl\")");

    // Array types for wrappers
    jl_value_t *array_float32 = jl_apply_array_type((jl_value_t*)jl_float32_type, 1);
    jl_value_t *array_int32 = jl_apply_array_type((jl_value_t*)jl_int32_type, 1);
    
    // Module
    jl_module_t *juliaKernelsModule = (jl_module_t*)jl_eval_string("JuliaKernels");
    
    // f_kernel_map_entry, f_kernel_map_projection and f_kernel_map_exit from Module GateKernels
    JL_GC_PUSH1(&juliaKernelsModule);
    jl_function_t *call_all = (jl_function_t*)jl_get_global(juliaKernelsModule, jl_symbol("call_all"));
    JL_GC_POP();

    // Wrappers
    // px, py, pz
    jl_array_t *jl_px = jl_ptr_to_array_1d(array_float32, particle->px, particle_size, 0);
    jl_array_t *jl_py = jl_ptr_to_array_1d(array_float32, particle->py, particle_size, 0);
    jl_array_t *jl_pz = jl_ptr_to_array_1d(array_float32, particle->pz, particle_size, 0);

    // dx, dy, dz
    jl_array_t *jl_dx = jl_ptr_to_array_1d(array_float32, particle->dx, particle_size, 0);
    jl_array_t *jl_dy = jl_ptr_to_array_1d(array_float32, particle->dy, particle_size, 0);
    jl_array_t *jl_dz = jl_ptr_to_array_1d(array_float32, particle->dz, particle_size, 0);

    // entry_collim_y entry_collim_z
    jl_array_t *jl_entry_collim_y = jl_ptr_to_array_1d(array_float32, collimator->entry_collim_y, y_size, 0);
    jl_array_t *jl_entry_collim_z = jl_ptr_to_array_1d(array_float32, collimator->entry_collim_z, z_size, 0);
    
    // exit_collim_y exit_collim_z
    jl_array_t *jl_exit_collim_y = jl_ptr_to_array_1d(array_float32, collimator->exit_collim_y, y_size, 0);
    jl_array_t *jl_exit_collim_z = jl_ptr_to_array_1d(array_float32, collimator->exit_collim_z, z_size, 0);

    // hole
    jl_array_t *jl_hole = jl_ptr_to_array_1d(array_int32, hole, particle_size, 0);

    // Args call_all
    jl_value_t **args;
    JL_GC_PUSHARGS(args,17);
    args[0] = (jl_value_t*)jl_px;
    args[1] = (jl_value_t*)jl_py;
    args[2] = (jl_value_t*)jl_pz;
    args[3] = (jl_value_t*)jl_dx;
    args[4] = (jl_value_t*)jl_dy;
    args[5] = (jl_value_t*)jl_dz;
    args[6] = (jl_value_t*)jl_entry_collim_y;
    args[7] = (jl_value_t*)jl_entry_collim_z;
    args[8] = (jl_value_t*)jl_exit_collim_y;
    args[9] = (jl_value_t*)jl_exit_collim_z;
    args[10] = (jl_value_t*)jl_hole;
    args[11] = jl_box_uint32(y_size);
    args[12] = jl_box_uint32(z_size);
    args[13] = jl_box_float32(planeToProject);
    args[14] = jl_box_int32(particle_size);
    args[15] = jl_box_int32(grid_size);
    args[16] = jl_box_int32(block_size);
    
    // Call call_all
    struct RetProjection *retproj = (RetProjection *)jl_call(call_all,args,17);

    // Accessing result data
    hole = (int*)jl_array_data(retproj->hole);

    particle->px = (float*)jl_array_data(retproj->px);
    particle->py = (float*)jl_array_data(retproj->py);
    particle->pz = (float*)jl_array_data(retproj->pz);

    JL_GC_POP();
    jl_atexit_hook(0);

    // Pack data to CPU
    int c = 0;
    int i = 0;
    while( i < particle_size )
    {
        if( hole[ i ] == -1 )
        {
            ++i;
            continue;
        }
        //h_hole[ c ] = h_hole[ i ];
        particle->px[ c ] = particle->px[ i ];
        particle->py[ c ] = particle->py[ i ];
        particle->pz[ c ] = particle->pz[ i ];
        particle->dx[ c ] = particle->dx[ i ];
        particle->dy[ c ] = particle->dy[ i ];
        particle->dz[ c ] = particle->dz[ i ];
                particle->eventID[ c ] = particle->eventID[ i ];
                particle->parentID[ c ] = particle->parentID[ i ];
                particle->trackID[ c ] = particle->trackID[ i ];
                particle->t[ c ] = particle->t[ i ];
                particle->E[ c ] = particle->E[ i ];
                particle->type[ c ] = particle->type[ i ];
        ++c;
        ++i;
    }
    particle->size = c;    

    free(hole);
}