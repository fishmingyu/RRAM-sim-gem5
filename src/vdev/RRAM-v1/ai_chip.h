#pragma once
#include "core.h"

class AIChip
{
public:
	int tickPerCycle = 1;

	int run();

	void *mem1;
	void *mem2;
	void *mem3;

	bool finished = false;
	unsigned char counter = 0;

	AIChip()
	{
		core.memhub.addMemory("activation_mem1", 1024);
		core.memhub.addMemory("activation_mem2", 1024 * 5);
		core.memhub.addMemory("weight_and_bias_mem", 3145728);

		mem1 = core.memhub.getMemory("activation_mem1").getPtr();
		mem2 = core.memhub.getMemory("activation_mem2").getPtr();
		mem3 = core.memhub.getMemory("weight_and_bias_mem").getPtr();
	}

private:
	Core core;
};