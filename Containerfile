FROM libmpopt_base

COPY --chown=user:user . libmpopt

ARG BUILDTYPE=debugoptimized

RUN mkdir libmpopt-build \
&& cd libmpopt-build \
&& meson setup \
    -Db_ndebug=if-release \
    -Dbuildtype=${BUILDTYPE} \
    -Dqpbo=enabled \
    -Dgurobi=auto \
    /home/user/libmpopt \
&& ninja \
&& sudo meson install \
&& sudo ldconfig

# vim: set ts=8 sts=4 sw=4 et ft=dockerfile:
