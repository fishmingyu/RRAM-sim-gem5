## RRAM simulator with gem5. 
We use three layers' MLP to simulate the RRAM's performance.
RRAM peripheral code is in src/vdev
RRAM-v1 is an RRAM without noise and load weight for every layer
RRAM-v2 is an RRAM that load weight once

simulation code is in tests/test-progs/simplemlp

To bulid gem5
``` scons build/ARM/gem5.opt -j8 ```
To run simulation
``` build/ARM/gem5.opt configs/example/simple_mlp.py ```

you could enter the file and change your simluation code
``` cd tests/test-progs/simplemlp ```
edit main.c then ```make```