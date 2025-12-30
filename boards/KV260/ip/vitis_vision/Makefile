# Copyright (C) 2021-2022 Xilinx, Inc
# Copyright (C) 2023-2025 Advanced Micro Devices, Inc.

# SPDX-License-Identifier: BSD-3-Clause

.PHONY: all

HLS_DIR = $(shell readlink -f ../../../../src/)
SRC_DIRS  = $(shell ls $(HLS_DIR))
HLS_PROJECTS = $(addsuffix  .vhlsprj, $(SRC_DIRS))

all: submodule $(HLS_PROJECTS)

DEVICE := zynq
FREQ := 156.25
FPGAPART := xc7z020clg400-1
DEVICE_MACRO := _ZYNQ_
VITIS_LIB_INCLUDE = $(shell readlink -f ../../../../Vitis_Libraries/vision/L1/include/)


submodule:
	cd ../../../../ && git submodule init && git submodule update

ifeq ($(DEVICE),zynq-us)
FPGAPART = xczu5eg-sfvc784-1-e
FREQ = 300
DEVICE_MACRO = _ZYNQ_US_
else ifeq ($(DEVICE),zynq-us-no-uram)
FPGAPART = xczu3eg-sfvc784-2-e
FREQ = 300
DEVICE_MACRO = _ZYNQ_US_NO_URAM_
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

	v++ -c --mode hls --config $(subst .vhlsprj,,$@)_hls.cfg --work_dir $(subst .vhlsprj,,$@).vhlsprj
	vitis-run --mode hls --package --config $(subst .vhlsprj,,$@)_hls.cfg --work_dir $(subst .vhlsprj,,$@).vhlsprj

clean:
	rm -rf *.log *.jou *.str *.cfg _ide/ *.json logs/

distclean: clean
	rm -rf *.vhlsprj
