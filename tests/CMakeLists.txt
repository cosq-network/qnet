add_executable(test_tensor test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE qnet)
add_test(NAME test_tensor COMMAND test_tensor)

add_executable(test_graph test_graph.cpp)
target_link_libraries(test_graph PRIVATE qnet)
add_test(NAME test_graph COMMAND test_graph)

add_executable(test_safetensors test_safetensors.cpp)
target_link_libraries(test_safetensors PRIVATE qnet)
add_test(NAME test_safetensors COMMAND test_safetensors)

add_executable(test_conv test_conv.cpp)
target_link_libraries(test_conv PRIVATE qnet)
add_test(NAME test_conv COMMAND test_conv)

add_executable(test_serializer test_serializer.cpp)
target_link_libraries(test_serializer PRIVATE qnet)
add_test(NAME test_serializer COMMAND test_serializer)

add_executable(test_backward test_backward.cpp)
target_link_libraries(test_backward PRIVATE qnet)
add_test(NAME test_backward COMMAND test_backward)
