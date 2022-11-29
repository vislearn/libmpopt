project('libmpopt', ['c', 'cpp'], version: '0.1.0', default_options: ['cpp_std=c++17'])

#
# Project specific infos.
#

sublibraries = ['ct', 'gm', 'qap']

python_wrappers = {
  'ct_jug':        'mpopt.ct.cmdline.jug',
  'ct_jug_2ts':    'mpopt.ct.cmdline.jug2_ts',
  'ct_jug_gurobi': 'mpopt.ct.cmdline.jug_gurobi',
  'ct_test':       'mpopt.ct.cmdline.test',
  'gm_uai':        'mpopt.gm.cmdline.uai',
  'dd_to_uai':     'mpopt.qap.cmdline.dd_to_uai',
  'qap_dd':        'mpopt.qap.cmdline.dd',
  'qap_dd_gurobi': 'mpopt.qap.cmdline.dd_gurobi',
}

python_sources = [
  'mpopt/common/gurobi.py',
  'mpopt/common/solver.py',
  'mpopt/ct/__init__.py',
  'mpopt/ct/cmdline/__init__.py',
  'mpopt/ct/cmdline/jug.py',
  'mpopt/ct/cmdline/jug_2ts.py',
  'mpopt/ct/cmdline/jug_gurobi.py',
  'mpopt/ct/cmdline/test.py',
  'mpopt/ct/gurobi.py',
  'mpopt/ct/jug.py',
  'mpopt/ct/model.py',
  'mpopt/ct/primals.py',
  'mpopt/ct/rounding.py',
  'mpopt/ct/tracker.py',
  'mpopt/gm/__init__.py',
  'mpopt/gm/cmdline/__init__.py',
  'mpopt/gm/cmdline/uai.py',
  'mpopt/gm/model.py',
  'mpopt/gm/solver.py',
  'mpopt/gm/uai.py',
  'mpopt/qap/__init__.py',
  'mpopt/qap/cmdline/__init__.py',
  'mpopt/qap/cmdline/dd.py',
  'mpopt/qap/cmdline/dd_gurobi.py',
  'mpopt/qap/cmdline/dd_to_uai.py',
  'mpopt/qap/dd.py',
  'mpopt/qap/gurobi.py',
  'mpopt/qap/model.py',
  'mpopt/qap/primals.py',
  'mpopt/qap/solver.py',
  'mpopt/utils.py']

#
# Features and Options.
#

prefer_static = get_option('prefer_static')

feature_gurobi = get_option('gurobi')
feature_qpbo = get_option('qpbo')

#
# Dependencies.
#

fs = import('fs')
gurobi = dependency('gurobi-cpp', required: feature_gurobi)
python3 = import('python').find_installation('python3')
qpbo = dependency('libqpbo', required: feature_qpbo)
swig = find_program('swig', required: true)

#
# Configuration Data.
#

configuration = configuration_data()

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

subdir('include/mpopt/common')
swig_include_dirs = ['-I' + meson.source_root() / 'include', '-I' + meson.build_root() / 'include']
include_dirs = include_directories('include')

#
# C++ libraries and Python bindings.
#

foreach sublibrary : sublibraries
  lib = both_libraries(
    'mpopt_' + sublibrary, 'lib' / sublibrary + '.cpp',
    include_directories: include_dirs,
    version: meson.project_version(),
    soversion: '0',
    pic: true,
    dependencies: [gurobi, qpbo],
    install: true)

  if prefer_static
    lib = lib.get_static_lib()
  endif

  lib_py = custom_target('lib' + sublibrary + '_py',
    input: ['swig/' + sublibrary + '.i', 'include/mpopt/' + sublibrary + '.h'],
    output: ['lib' + sublibrary + '_py.c'],
    command: [swig, '-python', '-noproxy', swig_include_dirs, '-o', '@OUTPUT@', '@INPUT0@'])

  python3.extension_module('libmpopt_' + sublibrary,
    sources: [lib_py],
    include_directories: include_dirs,
    link_with: [lib],
    dependencies: [python3.dependency()],
    install: true,
    install_dir: python3.get_install_dir(pure: false, subdir: 'mpopt' / sublibrary))

  install_headers('include/mpopt/' + sublibrary + '.h', subdir: 'mpopt')
endforeach

#
# Python Sources.
#

foreach python_source : python_sources
  src = 'python' / python_source
  dir = fs.parent(python_source)
  python3.install_sources([src], pure: true, subdir: dir)
endforeach

configure_file(
  input: 'python/mpopt/common/config.py.in',
  output: 'config.py',
  configuration: configuration,
  install: true,
  install_dir: python3.get_install_dir() / 'mpopt/common')

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
