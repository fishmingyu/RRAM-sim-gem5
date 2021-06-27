#include <cassert>
#include <algorithm>
#include <cstring>
#include <memory>
#include <math.h>

#include "core.h"
#include "rram_array.h"

int Core::op_init(void)
{
	/*set up memories*/
	//create weight & bias mem
	float *mem3 = memhub.getMemory("weight_and_bias_mem").getPtr();

	memhub.addSharedMemory("model/fc_1/weight", mem3, 1024, 1024 / 4);
	mem3 += 1024 * 1024 / 4;
	memhub.addSharedMemory("model/bn_1/bias", mem3, 1024);
	mem3 += 1024;
	memhub.addSharedMemory("model/bn_1/mean", mem3, 1024);
	mem3 += 1024;
	memhub.addSharedMemory("model/bn_1/var", mem3, 1024);
	mem3 += 1024;
	memhub.addSharedMemory("model/bn_1/weight", mem3, 1024);
	mem3 += 1024;

	memhub.addSharedMemory("model/fc_2/weight", mem3, 1024, 1024 / 4);
	mem3 += 1024 * 1024 / 4;
	memhub.addSharedMemory("model/bn_2/bias", mem3, 1024);
	mem3 += 1024;
	memhub.addSharedMemory("model/bn_2/mean", mem3, 1024);
	mem3 += 1024;
	memhub.addSharedMemory("model/bn_2/var", mem3, 1024);
	mem3 += 1024;
	memhub.addSharedMemory("model/bn_2/weight", mem3, 1024);
	mem3 += 1024;

	memhub.addSharedMemory("model/fc_3/weight", mem3, 1024, 1024 / 4);
	mem3 += 1024 * 1024 / 4;
	memhub.addSharedMemory("model/bn_3/bias", mem3, 1024);
	mem3 += 1024;
	memhub.addSharedMemory("model/bn_3/mean", mem3, 1024);
	mem3 += 1024;
	memhub.addSharedMemory("model/bn_3/var", mem3, 1024);
	mem3 += 1024;
	memhub.addSharedMemory("model/bn_3/weight", mem3, 1024);
	mem3 += 1024;

	//create activation data memory mapping
	float *mem1 = memhub.getMemory("activation_mem1").getPtr();
	float *mem2 = memhub.getMemory("activation_mem2").getPtr();

	memhub.addSharedMemory("data/fc_1/in", mem1, 1024);
	memhub.addSharedMemory("data/fc_1/out", mem2, 1024);
	memhub.addSharedMemory("data/fc_2/in", mem1, 1024);
	memhub.addSharedMemory("data/fc_2/out", mem2, 1024);
	memhub.addSharedMemory("data/fc_3/in", mem1, 1024);
	memhub.addSharedMemory("data/fc_3/out", mem2, 1024);

	return 1;
}

int Core::op_input(std::string fromMemName, std::string toMemName)
{
	int8_t *from = (int8_t *)memhub.getMemory(fromMemName).getPtr();
	int8_t *to = (int8_t *)memhub.getMemory(toMemName).getPtr();
	for (int x = 0; x < 28; x++)
	{
		for (int y = 0; y < 28; y++)
		{
			to[x * 28 + y] = from[x * 28 + y];
		}
	}
	return 10;
}

int Core::op_output(std::string memName)
{
	float *ptr = memhub.getMemory(memName).getPtr();
	outputIndex = std::max_element(ptr, ptr + 10) - ptr;
	return 10;
}

int Core::op_load(
	std::string weightMemName,
	std::string bnBiasName,
	std::string bnMeanName,
	std::string bnVarName,
	std::string bnWeightName)
{
	Memory &weight = memhub.getMemory(weightMemName);
	Memory &batchnorm_bias = memhub.getMemory(bnBiasName);
	Memory &batchnorm_mean = memhub.getMemory(bnMeanName);
	Memory &batchnorm_var = memhub.getMemory(bnVarName);
	Memory &batchnorm_weight = memhub.getMemory(bnWeightName);

	return Rarray.load_weight((std::int8_t(*)[1024])weight.getPtr(), batchnorm_mean.getPtr(), batchnorm_var.getPtr(), batchnorm_weight.getPtr(), batchnorm_bias.getPtr());
}

int Core::op_calculate(
	std::string inMemName,
	std::string outMemName)
{
	Memory &input = memhub.getMemory(inMemName);
	Memory &output = memhub.getMemory(outMemName);
	return Rarray.calculate((std::int8_t *)input.getPtr(), output.getPtr());
}

int Core::op_sign(
	std::string inMemName,
	std::string outMemName)
{
	Memory &input = memhub.getMemory(inMemName);
	Memory &output = memhub.getMemory(outMemName);
	return Rarray.sign(input.getPtr(), (std::int8_t *)output.getPtr());
}