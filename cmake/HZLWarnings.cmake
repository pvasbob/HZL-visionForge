function(hzl_enable_warnings target_name)
    if(MSVC)
        target_compile_options(
            ${target_name}
            INTERFACE
                $<$<COMPILE_LANGUAGE:CXX>:/W4;/permissive->
        )
    else()
        target_compile_options(
            ${target_name}
            INTERFACE
                $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Wconversion;-Wshadow>
        )
    endif()
endfunction()
