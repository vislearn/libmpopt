%module libmpopt_mwis_original
%{
  #include <mpopt/mwis_original.h>
%}

%typemap(in) int* {
  $1 = (int*)(PyInt_AsLong($input));
}

%rename ("%(strip:[mpopt_mwis_])s") "";
%include <mpopt/mwis_original.h>
