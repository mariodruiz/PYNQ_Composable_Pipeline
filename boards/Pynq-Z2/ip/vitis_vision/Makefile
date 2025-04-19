# Copyright (C) 2021-2025 Xilinx, Inc

# SPDX-License-Identifier: BSD-3-Clause

.PHONY: all

HLS_DIR = $(shell readlink -f ../../../../src/)
SRC_DIRS  = $(shell ls $(HLS_DIR))
HLS_PROJECTS = $(addsuffix  .vhlsprj, $(SRC_DIRS))

all: submodule $(HLS_PROJECTS)

DEVICE ?= zynq
FREQ ?= 156.25
FPGAPART ?= xc7z020clg400-1
DEVICE_MACRO := _ZYNQ_
VITIS_LIB_INCLUDE = $(shell readlink -f ../../../../Vitis_Libraries/vision/L1/include/)


submodule:
	cd ../../../../ && git submodule init && git submodule update

ifeq ($(DEVICE),zynq-us)
	FPGAPART = xczu5eg-sfvc784-1-e
	FREQ = 300
	DEVICE_MACRO = _ZYNQ_US_
endif

%.vhlsprj:
	@echo "Generating config file for project: $*"
	@proj_name=$(subst .vhlsprj,,$@); \
	config_file_name="$${proj_name}_hls.cfg"; \
	echo "part=$(FPGAPART)"; \
	echo "[hls]"; \
	echo "flow_target=vivado"; \
	echo "package.output.format=ip_catalog"; \
	echo "package.output.syn=false"; \
	echo "syn.top=$${proj_name}_accel"; \
	echo "syn.file=$(HLS_DIR)/$${proj_name}/$${proj_name}.cpp"; \
	echo "syn.file_cflags=I$(VITIS_LIB_INCLUDE) -D${DEVICE_MACRO}"; \
	echo "clock=$(FREQ)"; \

	v++ -c --mode hls --config $(config_file_name) --work_dir ip
	vitis-run --mode hls --package --config $(config_file_name) --work_dir ip

clean:
	rm -rf *.log *.jou *.str .cfg _ide/

distclean: clean
	rm -rf *.vhlsprj
