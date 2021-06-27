rm -rf build
scons build/ARM/gem5.opt -j40
build/ARM/gem5.opt configs/example/simple_cnn.py