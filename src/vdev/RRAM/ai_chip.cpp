#include "ai_chip.h"

int AIChip::run()
{
	int cyclesUsed = 0;
	finished = false;

	if (counter == 0)
	{
		cyclesUsed = core.op_init();
	}
	//load
	else if (counter == 1)
	{
		cyclesUsed = core.op_load(
			"model/fc_1/weight",
			"model/bn_1/bias", "model/bn_1/mean",
			"model/bn_1/var", "model/bn_1/weight",
			"model/fc_2/weight",
			"model/bn_2/bias", "model/bn_2/mean",
			"model/bn_2/var", "model/bn_2/weight",
			"model/fc_3/weight",
			"model/bn_3/bias", "model/bn_3/mean",
			"model/bn_3/var", "model/bn_3/weight");
	}
	else if (counter == 2)
	{
		cyclesUsed = core.op_input("activation_mem2", "data/fc_1/in");
	}
	//fc1 and bn1
	else if (counter == 3)
	{
		cyclesUsed = core.op_calculate(
			"data/fc_1/in", "data/fc_1/out", 1);
	}
	//sign 1
	else if (counter == 4)
	{
		cyclesUsed = core.op_sign(
			"data/fc_1/out", "data/fc_2/in");
	}
	//fc2 and bn2
	else if (counter == 5)
	{
		cyclesUsed = core.op_calculate(
			"data/fc_2/in", "data/fc_2/out", 2);
	}
	//sign 2
	else if (counter == 6)
	{
		cyclesUsed = core.op_sign(
			"data/fc_2/out", "data/fc_3/in");
	}
	//fc3 and bn3
	else if (counter == 7)
	{
		cyclesUsed = core.op_calculate(
			"data/fc_3/in", "data/fc_3/out", 3);
	}
	//output
	else if (counter == 8)
	{
		cyclesUsed = core.op_output("data/fc_3/out");
		*(int8_t *)(mem2) = core.outputIndex;
		finished = true;
	}

	return cyclesUsed * tickPerCycle;
}
