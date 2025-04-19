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
	{ \
	echo "part=$(FPGAPART)"; \
	echo ""; \
	echo "[hls]"; \
	echo "clock=$(FREQ)"; \
	echo "syn.top=$${proj_name}_accel"; \
	echo "flow_target=vivado"; \
	echo "package.output.format=ip_catalog"; \
	echo "package.output.syn=false"; \
	echo "syn.file=$(HLS_DIR)/$${proj_name}/$${proj_name}.cpp"; \
	echo "syn.cflags=-I$(VITIS_LIB_INCLUDE) -D${DEVICE_MACRO}"; \
	} > "$${config_file_name}"

	v++ -c --mode hls --config $(subst .vhlsprj,,$@)_hls.cfg --work_dir ip_$(subst .vhlsprj,,$@)
	vitis-run --mode hls --package --config $(subst .vhlsprj,,$@)_hls.cfg --work_dir ip_$(subst .vhlsprj,,$@)

clean:
	rm -rf *.log *.jou *.str *.cfg _ide/ *.json

distclean: clean
	rm -rf *.vhlsprj ip_*/
