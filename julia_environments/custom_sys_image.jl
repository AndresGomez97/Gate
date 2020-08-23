Base.reinit_stdio()
Base.init_depot_path()
Base.init_load_path()

using CUDA

empty!(LOAD_PATH)
empty!(DEPOT_PATH)
