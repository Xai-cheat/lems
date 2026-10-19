cmake_minimum_required(VERSION 3.22.1)
project(retrolens)
add_library(retrolens SHARED filters.cpp)
# debug builds default to -O0, which makes the per-pixel filter very slow
target_compile_options(retrolens PRIVATE -O3 -ffast-math)
find_library(log-lib log)
target_link_libraries(retrolens jnigraphics ${log-lib})
