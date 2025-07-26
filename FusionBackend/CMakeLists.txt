cmake_minimum_required(VERSION 3.8)

project(FusionBackend)

if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

set(PYTHON_RELEASE_LIB "C:/Users/arthu/AppData/Local/Programs/Python/Python313/libs/python313.lib")

add_library(FusionBackend MODULE
  "FusionBackend.cpp"
 )

target_include_directories(FusionBackend PRIVATE ${Python3_INCLUDE_DIRS})

target_link_options(FusionBackend PRIVATE 
    "$<$<CONFIG:Debug>:/NODEFAULTLIB:python313_d.lib>"
    "/EXPORT:PyInit_Fusion"
)

target_link_libraries(FusionBackend PRIVATE "${PYTHON_RELEASE_LIB}")

set_target_properties(FusionBackend PROPERTIES
    PREFIX ""
    SUFFIX ".pyd"
    OUTPUT_NAME "Fusion"
    RUNTIME_OUTPUT_DIRECTORY         "${CMAKE_SOURCE_DIR}/../Fusion"
    LIBRARY_OUTPUT_DIRECTORY         "${CMAKE_SOURCE_DIR}/../Fusion"
    ARCHIVE_OUTPUT_DIRECTORY         "${CMAKE_SOURCE_DIR}/../Fusion"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG  "${CMAKE_SOURCE_DIR}/../Fusion"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG  "${CMAKE_SOURCE_DIR}/../Fusion"
    ARCHIVE_OUTPUT_DIRECTORY_DEBUG  "${CMAKE_SOURCE_DIR}/../Fusion"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE  "${CMAKE_SOURCE_DIR}/../Fusion"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE  "${CMAKE_SOURCE_DIR}/../Fusion"
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE  "${CMAKE_SOURCE_DIR}/../Fusion"
)

target_compile_features(FusionBackend PRIVATE cxx_std_20)