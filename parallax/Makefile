# Parallax build. C11 + libc/libm/pthreads only. No third-party code, ever.
#
# Layout:  src/<module>/*.c   -> $(BUILD)/lib/libplx_<module>.a
#          tests/test_<x>.c   -> $(BUILD)/bin/test_<x>
#          apps/<x>.c         -> $(BUILD)/bin/<x>
#
# A program declares the modules it links with one comment line in its source:
#     // plx-deps: feat ops img core
# so adding a test or app never requires touching this file.
#
#   make                 build every app
#   make test            build + run every test
#   make test-la         build + run tests/test_la.c only (builds only its deps)
#   make BUILD=build-x   use a private build dir (parallel work in one tree)
#   make SAN=1           AddressSanitizer + UBSan build (use its own BUILD dir)

SHELL := /bin/bash
CC    ?= cc
BUILD ?= build
OPT   ?= -O3 -march=native

CFLAGS := -std=c11 $(OPT) -g -Wall -Wextra -Wshadow -Wno-unused-parameter \
          -D_POSIX_C_SOURCE=200809L -D_DEFAULT_SOURCE -Isrc -pthread $(EXTRA_CFLAGS)
LDLIBS := -lm -pthread
ifeq ($(SAN),1)
CFLAGS += -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=undefined
endif

MODULES := $(notdir $(wildcard src/*))
TESTS   := $(patsubst tests/%.c,%,$(wildcard tests/test_*.c))
APPS    := $(patsubst apps/%.c,%,$(wildcard apps/*.c))

.PHONY: all test clean
all: $(addprefix $(BUILD)/bin/,$(APPS))

$(BUILD)/obj/%.o: src/%.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

define MODULE_RULE
$(BUILD)/lib/libplx_$(1).a: $$(patsubst src/%.c,$(BUILD)/obj/%.o,$$(wildcard src/$(1)/*.c))
	@mkdir -p $$(@D)
	@rm -f $$@
	$(AR) rcs $$@ $$^
endef
$(foreach m,$(MODULES),$(eval $(call MODULE_RULE,$(m))))

plxdeps = $(shell sed -n 's|^// plx-deps:||p' $(1))
plxlibs = $(foreach m,$(call plxdeps,$(1)),$(BUILD)/lib/libplx_$(m).a)

# $(1) = source file, $(2) = output binary
define PROG_RULE
$(2): $(1) $(call plxlibs,$(1)) $(wildcard tests/*.h)
	@mkdir -p $$(@D)
	$(CC) $(CFLAGS) -Itests $(1) -Wl,--start-group $(call plxlibs,$(1)) -Wl,--end-group $(LDLIBS) -o $$@
endef
$(foreach t,$(TESTS),$(eval $(call PROG_RULE,tests/$(t).c,$(BUILD)/bin/$(t))))
$(foreach a,$(APPS),$(eval $(call PROG_RULE,apps/$(a).c,$(BUILD)/bin/$(a))))

test: $(addprefix $(BUILD)/bin/,$(TESTS))
	@fail=0; for t in $^; do echo "== $$t"; $$t || fail=1; done; exit $$fail

test-%: $(BUILD)/bin/test_%
	$<

clean:
	rm -rf $(BUILD)

-include $(shell find $(BUILD)/obj -name '*.d' 2>/dev/null)
