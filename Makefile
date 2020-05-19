export PATH := /usr/local/cuda-9.0/bin:$(PATH)
export LD_LIBRARY_PATH := /usr/local/cuda-9.0/lib64:$(LD_LIBRARY_PATH)
export CUDA_VISIBLE_DEVICES := 0

all: draw_rectangles box_intersections nms roi_align

draw_rectangles:
	cd lib/draw_rectangles; python setup.py build_ext --inplace
box_intersections:
	cd lib/fpn/box_intersections_cpu; python setup.py build_ext --inplace
nms:
	cd lib/fpn/nms; make
roi_align:
	cd lib/fpn/roi_align; make
