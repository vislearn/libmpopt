%module libmpopt_mwis_temp_cont
%{
  #include <mpopt/mwis_temp_cont.h>
%}

%typemap(in) int* {
  $1 = (int*)(PyInt_AsLong($input));
}

%rename ("%(strip:[mpopt_mwis_])s") "";
%include <mpopt/mwis_temp_cont.h>
