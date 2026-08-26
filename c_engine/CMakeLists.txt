cmake_minimum_required(VERSION 3.15)
project(tank_pdm_infer C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# -Wconversion is informative but not fatal: it flags a great deal of benign
# int/float index arithmetic in tight numeric loops. Correctness-critical
# warnings are errors; style warnings are not.
if(MSVC)
  add_compile_options(/W4 /WX)
else()
  add_compile_options(-Wall -Wextra -Werror -Wshadow -Wconversion -Wno-error=conversion)
endif()

add_library(tank_pdm_infer_shared SHARED tank_pdm_infer.c tank_pdm_weights.c)
set_target_properties(tank_pdm_infer_shared PROPERTIES
                      OUTPUT_NAME tank_pdm_infer
                      PREFIX "lib")

add_executable(tank_pdm_selftest main_test.c tank_pdm_infer.c tank_pdm_weights.c)

if(NOT MSVC)
  target_link_libraries(tank_pdm_infer_shared m)
  target_link_libraries(tank_pdm_selftest m)
endif()
