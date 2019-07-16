%module libmpopt_gm
%{
  #include <mpopt/gm.h>
%}

%rename ("%(strip:[mpopt_gm_])s") "";
%include <mpopt/gm.h>
