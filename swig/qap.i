%module libmpopt_qap
%{
  #include <mpopt/qap.h>
%}

%rename ("%(strip:[mpopt_qap_])s") "";
%include <mpopt/qap.h>
