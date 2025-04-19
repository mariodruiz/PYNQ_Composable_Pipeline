# Copyright (C) 2021-2025 Xilinx, Inc

# SPDX-License-Identifier: BSD-3-Clause

.PHONY: all

DIRECTORIES  = $(shell ls ../../../../src)
HLS_PROJECTS = $(addsuffix  .vhlsprj, $(DIRECTORIES))

all: submodule $(HLS_PROJECTS)

DEVICE ?= zynq
FREQ ?= 156.25
FPGAPART ?= xc7z020clg400-1
DEVICE_MACRO := _ZYNQ_
VITIS_LIB_INCLUDE = $(shell readlink -f ../../../../Vitis_Libraries/vision/L1/include/)
HLS_DIR = $(shell readlink -f ../../../../src/)


submodule:
	cd ../../../../ && git submodule init && git submodule update

ifeq ($(DEVICE),zynq-us)
	FPGAPART = xczu5eg-sfvc784-1-e
	FREQ = 300
	DEVICE_MACRO = _ZYNQ_US_
endif

%.vhlsprj:
	# write config file
	proj_name=$(subst .vhlsprj,,$@)
	config_file_name = $(subst .vhlsprj,,$@)_hls.conf
	echo "part=$(FPGAPART)" > $(config_file_name)
	echo "[hls]" >> $(config_file_name)
	echo "flow_target=vivado" >> $(config_file_name)
	echo "package.output.format=ip_catalog" >> $(config_file_name)
	echo "package.output.syn=false" >> $(config_file_name)
	echo "syn.top=$(proj_name)_accel" >> $(config_file_name)
	echo "syn.file=$(HLS_DIR)/$(proj_name)/$(proj_name).cpp" >> $(config_file_name)
	echo "syn.file_cflags=I$(VITIS_LIB_INCLUDE) -D${DEVICE_MACRO}" >> $(config_file_name)
	echo "clock=$(FREQ)" >> $(config_file_name)

	v++ -c --mode hls --config $(config_file_name) --work_dir ip
	vitis-run --mode hls --package --config $(config_file_name) --work_dir ip

clean:
	rm -rf *.log *.jou *.str

distclean: clean
	rm -rf *.vhlsprj
