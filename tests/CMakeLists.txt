# Set PATH so the test executables can find qnet.dll on Windows
if(WIN32)
    set(QNET_DLL_DIR "$<TARGET_FILE_DIR:qnet>")
    set(TEST_ENV "PATH=${QNET_DLL_DIR}\\;$ENV{PATH}")
else()
    set(TEST_ENV "")
endif()

function(add_qnet_test NAME SOURCE)
    add_executable(${NAME} ${SOURCE})
    target_link_libraries(${NAME} PRIVATE qnet)
    add_test(NAME ${NAME} COMMAND ${NAME})
    if(WIN32)
        set_tests_properties(${NAME} PROPERTIES ENVIRONMENT "${TEST_ENV}")
        add_custom_command(TARGET ${NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:qnet>"
                "$<TARGET_FILE_DIR:${NAME}>/"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE_DIR:qnet>/openblas.dll"
                "$<TARGET_FILE_DIR:${NAME}>/" )
    endif()
endfunction()

add_qnet_test(test_tensor test_tensor.cpp)
add_qnet_test(test_graph test_graph.cpp)
add_qnet_test(test_safetensors test_safetensors.cpp)
add_qnet_test(test_conv test_conv.cpp)
add_qnet_test(test_serializer test_serializer.cpp)
add_qnet_test(test_backward test_backward.cpp)
