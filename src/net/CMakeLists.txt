add_library(Net STATIC Net.cpp)
target_include_directories(Net PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(Net PUBLIC Matrix)