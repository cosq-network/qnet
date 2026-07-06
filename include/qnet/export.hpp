#pragma once

#if defined(_WIN32) || defined(_WIN64)
    #ifdef qnet_EXPORTS
        #define QNET_API __declspec(dllexport)
    #else
        #define QNET_API __declspec(dllimport)
    #endif
#else
    #define QNET_API
#endif
