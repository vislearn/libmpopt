foreach wrapper, module : python_wrappers
  tmp_config = configuration_data()
  tmp_config.merge_from(configuration)
  tmp_config.set('MODULE', module)

  configure_file(
    input: 'pywrapper.in',
    output: wrapper,
    configuration: tmp_config,
    install: true,
    install_dir: get_option('bindir'),
    install_mode: 'rwxr-xr-x')
endforeach

# vim: set ts=8 sts=2 sw=2 et:
