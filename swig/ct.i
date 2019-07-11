%module libmpopt_ct
%{
  #include <mpopt/ct.h>
%}

%rename ("%(strip:[mpopt_ct_])s") "";
%include <mpopt/ct.h>
