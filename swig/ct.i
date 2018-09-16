%module libct
%{
  #include <ct/ct.h>
%}

%rename ("%(strip:[ct_])s") "";
%include <ct/ct.h>
