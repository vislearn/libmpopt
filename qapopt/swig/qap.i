%module libqapopt
%{
  #include <qapopt/qap.h>
%}

%rename ("%(strip:[qapopt_])s") "";

%typemap(in) int* {
  $1 = (int*)(PyInt_AsLong($input));
}

%include <qapopt/qap.h>
