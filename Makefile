# SPDX-License-Identifier: LGPL-2.1-or-later
# libane: make; make install PREFIX=/usr DESTDIR=...
# Needs the kernel's uAPI headers (linux-api-headers: <drm/drm.h>, <linux/sync_file.h>).
CC      ?= cc
CFLAGS  ?= -O2 -g
# what the build needs, kept apart so that CFLAGS given to make adds to it; apple_ane_accel.h
# includes "drm.h", as the kernel's uAPI DRM headers do
ANE_CFLAGS = -std=gnu11 -Wall -Wextra -fPIC -idirafter /usr/include/drm
PREFIX  ?= /usr/local
LIBDIR  ?= $(PREFIX)/lib
INCDIR  ?= $(PREFIX)/include
VERSION  = 0.1.0
SOVER    = 0
NAME     = ane

all: lib$(NAME).so.$(VERSION) lib$(NAME).a lib$(NAME).pc

$(NAME).o: $(NAME).c $(NAME).h apple_ane_accel.h
	$(CC) $(ANE_CFLAGS) $(CPPFLAGS) $(CFLAGS) -c -o $@ $(NAME).c

lib$(NAME).a: $(NAME).o
	$(AR) rcs $@ $^

lib$(NAME).so.$(VERSION): $(NAME).o
	$(CC) $(ANE_CFLAGS) $(CFLAGS) $(LDFLAGS) -shared -Wl,-soname,lib$(NAME).so.$(SOVER) -o $@ $^
	ln -sf $@ lib$(NAME).so.$(SOVER)
	ln -sf $@ lib$(NAME).so

lib$(NAME).pc: Makefile
	printf '%s\n' 'prefix=$(PREFIX)' 'libdir=$(LIBDIR)' 'includedir=$(INCDIR)' '' \
	  'Name: lib$(NAME)' 'Description: The Apple Neural Engine through the Linux accel driver apple-ane' \
	  'Version: $(VERSION)' 'Libs: -L$${libdir} -l$(NAME)' 'Cflags: -I$${includedir}' > $@

# make check [MODELS="dir..."] [SAN=1]: a model directory holds model.hwx and model.json
MODELS  ?=
ifdef SAN
TEST_CFLAGS = -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all
endif

test/test: test/test.c $(NAME).c $(NAME).h apple_ane_accel.h
	$(CC) $(ANE_CFLAGS) $(CPPFLAGS) $(CFLAGS) $(TEST_CFLAGS) $(LDFLAGS) -o $@ test/test.c $(NAME).c -lpthread

check: all test/test
	test/test $(MODELS)
	ANE_LIB=$(CURDIR)/lib$(NAME).so.$(VERSION) python3 test/test_ane.py $(firstword $(MODELS))

lint:
	clang-format --dry-run --Werror $(NAME).c $(NAME).h test/test.c macos/anehwx.m
	clang-tidy --quiet $(NAME).c test/test.c -- $(ANE_CFLAGS)
	ruff format --check python test
	ruff check python test

install: all
	install -d $(DESTDIR)$(LIBDIR)/pkgconfig $(DESTDIR)$(INCDIR)
	install -m755 lib$(NAME).so.$(VERSION) $(DESTDIR)$(LIBDIR)/
	ln -sf lib$(NAME).so.$(VERSION) $(DESTDIR)$(LIBDIR)/lib$(NAME).so.$(SOVER)
	ln -sf lib$(NAME).so.$(VERSION) $(DESTDIR)$(LIBDIR)/lib$(NAME).so
	install -m644 lib$(NAME).a $(DESTDIR)$(LIBDIR)/
	install -m644 lib$(NAME).pc $(DESTDIR)$(LIBDIR)/pkgconfig/
	install -m644 $(NAME).h $(DESTDIR)$(INCDIR)/

clean:
	rm -f *.o *.a *.so *.so.* *.pc test/test

.PHONY: all check lint install clean
