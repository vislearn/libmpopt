project('libmpopt', ['c', 'cpp'], version: '0.1.0', default_options: ['cpp_std=c++17'])

#
# Dependencies.
#

libqpbo_proj  = subproject('libqpbo')
libqpbo_dep   = libqpbo_proj.get_variable('libqpbo_dep')

#
# Include Directories.
#

configuration = configuration_data()
configuration.set('ENABLE_QPBO', true)
configuration.set10('ENABLE_QPBO_INT', true)
subdir('include/mpopt/common')

include_dirs = include_directories('include')

#
# C++ libraries
#
libmpopt = static_library(
                    'mpopt_qap', 
                    'lib/qap.cpp',
                    include_directories: include_dirs,
                    dependencies: libqpbo_dep,
                    install: false)

libmpopt_dep = declare_dependency(include_directories : include_dirs, link_with : libmpopt)
