%module libmpopt_mwis
%{
  #include <mpopt/mwis.h>
%}

%typemap(in) int* {
  $1 = (int*)(PyInt_AsLong($input));
}

%rename ("%(strip:[mpopt_mwis_])s") "";
%include <mpopt/mwis.h>
