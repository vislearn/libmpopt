project('qapopt', ['c', 'cpp'], version: '0.1.0', default_options: ['cpp_std=c++17'])

#
# Project specific infos.
#

python_wrappers = {
  'qap_dd':            'qapopt.qap.cmdline.dd',
  'qap_dd_fusion':     'qapopt.qap.cmdline.dd_fusion',
  'qap_dd_greedy_gen': 'qapopt.qap.cmdline.dd_greedy_gen',
}

python_sources = [
  'qapopt/common/gurobi.py',
  'qapopt/common/solver.py',
  'qapopt/qap/__init__.py',
  'qapopt/qap/cmdline/__init__.py',
  'qapopt/qap/cmdline/dd.py',
  'qapopt/qap/cmdline/dd_fusion.py',
  'qapopt/qap/cmdline/dd_greedy_gen.py',
  'qapopt/qap/dd.py',
  'qapopt/qap/gurobi.py',
  'qapopt/qap/model.py',
  'qapopt/qap/primals.py',
  'qapopt/qap/solver.py',
  'qapopt/utils.py']

#
# Features and Options.
#

feature_lsatr = get_option('lsatr')
feature_gurobi = get_option('gurobi')
feature_qpbo = get_option('qpbo')

#
# Dependencies.
#

fs = import('fs')
lsatr = dependency('liblsatr', required: feature_lsatr)
gurobi = dependency('gurobi-cpp', required: feature_gurobi)
python3 = import('python').find_installation('python3')
qpbo = dependency('libqpbo', required: feature_qpbo)
swig = find_program('swig', required: true)

#
# Configuration Data.
#

configuration = configuration_data()

configuration.set('ENABLE_LSATR', lsatr.found())
configuration.set10('ENABLE_LSATR_INT', lsatr.found())

configuration.set('ENABLE_GUROBI', gurobi.found())
configuration.set10('ENABLE_GUROBI_INT', gurobi.found())

configuration.set('ENABLE_QPBO', qpbo.found())
configuration.set10('ENABLE_QPBO_INT', qpbo.found())

configuration.set('PYTHON_EXECUTABLE', python3.path())
configuration.set('PYTHON_OPTIMIZEFLAG', '')
configuration.set('PYTHON_INSTALL_DIR', python3.get_install_dir())

if get_option('b_ndebug') == 'true' or get_option('b_ndebug') == 'if-release' and get_option('buildtype') == 'release'
  configuration.set('PYTHON_OPTIMIZEFLAG', '-OO')
endif

#
# Include Directories.
#

subdir('include/qapopt/common')
swig_include_dirs = ['-I' + meson.source_root() / 'include', '-I' + meson.build_root() / 'include']
include_dirs = include_directories('include')

#
# C++ libraries and Python bindings.
#

lib = both_libraries(
  'qapopt', 'lib/qap.cpp',
  include_directories: include_dirs,
  version: meson.project_version(),
  soversion: '0',
  pic: true,
  dependencies: [lsatr, gurobi, qpbo],
  install: true)

lib_py = custom_target('qapopt_py',
  input: ['swig/qap.i', 'include/qapopt/qap.h'],
  output: ['qapopt_py.c'],
  command: [swig, '-python', '-noproxy', swig_include_dirs, '-o', '@OUTPUT@', '@INPUT0@'])

python3.extension_module('libqapopt',
  sources: [lib_py],
  include_directories: include_dirs,
  link_with: [lib.get_static_lib()],
  dependencies: [python3.dependency()],
  install: true,
  install_dir: python3.get_install_dir(pure: false, subdir: 'qapopt/qap'))

install_headers('include/qapopt/qap.h')

#
# Python Sources.
#

foreach python_source : python_sources
  src = 'python' / python_source
  dir = fs.parent(python_source)
  python3.install_sources([src], pure: true, subdir: dir)
endforeach

configure_file(
  input: 'python/qapopt/common/config.py.in',
  output: 'config.py',
  configuration: configuration,
  install: true,
  install_dir: python3.get_install_dir() / 'qapopt/common')

#
# Python Exec Wrappers.
#

subdir('bin')

#
# Post Install Script.
#

postinst_script = configure_file(
  input: 'meson.postinst.in',
  output: 'meson.postinst',
  configuration: configuration)

meson.add_install_script(meson.current_build_dir() / 'meson.postinst')

# vim: set ts=8 sts=2 sw=2 et:
