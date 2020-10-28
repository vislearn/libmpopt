FROM libmpopt_base

COPY --chown=user:user . libmpopt

RUN mkdir libmpopt-build \
&& cd libmpopt-build \
&& meson setup \
    -Db_ndebug=if-release \
    -Dbuildtype=debugoptimized \
    -Dqpbo=enabled \
    /home/user/libmpopt \
&& ninja \
&& sudo meson install

# vim: set ts=8 sts=4 sw=4 et ft=dockerfile:
