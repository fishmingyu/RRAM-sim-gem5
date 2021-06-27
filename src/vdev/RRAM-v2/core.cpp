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
	std::string weight1MemName,
	std::string bnBias1Name,
	std::string bnMean1Name,
	std::string bnVar1Name,
	std::string bnWeight1Name,
	std::string weight2MemName,
	std::string bnBias2Name,
	std::string bnMean2Name,
	std::string bnVar2Name,
	std::string bnWeight2Name,
	std::string weight3MemName,
	std::string bnBias3Name,
	std::string bnMean3Name,
	std::string bnVar3Name,
	std::string bnWeight3Name)
{
	Memory &weight1 = memhub.getMemory(weight1MemName);
	Memory &batchnorm_bias1 = memhub.getMemory(bnBias1Name);
	Memory &batchnorm_mean1 = memhub.getMemory(bnMean1Name);
	Memory &batchnorm_var1 = memhub.getMemory(bnVar1Name);
	Memory &batchnorm_weight1 = memhub.getMemory(bnWeight1Name);
	Memory &weight2 = memhub.getMemory(weight2MemName);
	Memory &batchnorm_bias2 = memhub.getMemory(bnBias2Name);
	Memory &batchnorm_mean2 = memhub.getMemory(bnMean2Name);
	Memory &batchnorm_var2 = memhub.getMemory(bnVar2Name);
	Memory &batchnorm_weight2 = memhub.getMemory(bnWeight2Name);
	Memory &weight3 = memhub.getMemory(weight3MemName);
	Memory &batchnorm_bias3 = memhub.getMemory(bnBias3Name);
	Memory &batchnorm_mean3 = memhub.getMemory(bnMean3Name);
	Memory &batchnorm_var3 = memhub.getMemory(bnVar3Name);
	Memory &batchnorm_weight3 = memhub.getMemory(bnWeight3Name);

	return Rarray.load_weight((std::int8_t(*)[1024])weight1.getPtr(), (std::int8_t(*)[1024])weight2.getPtr(), (std::int8_t(*)[1024])weight3.getPtr(),
							  batchnorm_mean1.getPtr(), batchnorm_mean2.getPtr(), batchnorm_mean3.getPtr(),
							  batchnorm_var1.getPtr(), batchnorm_var2.getPtr(), batchnorm_var3.getPtr(),
							  batchnorm_weight1.getPtr(), batchnorm_weight2.getPtr(), batchnorm_weight3.getPtr(),
							  batchnorm_bias1.getPtr(), batchnorm_bias2.getPtr(), batchnorm_bias3.getPtr());
}

int Core::op_calculate(
	std::string inMemName,
	std::string outMemName,
	int fc)
{
	Memory &input = memhub.getMemory(inMemName);
	Memory &output = memhub.getMemory(outMemName);
	return Rarray.calculate((std::int8_t *)input.getPtr(), output.getPtr(), fc);
}

int Core::op_sign(
	std::string inMemName,
	std::string outMemName)
{
	Memory &input = memhub.getMemory(inMemName);
	Memory &output = memhub.getMemory(outMemName);
	return Rarray.sign(input.getPtr(), (std::int8_t *)output.getPtr());
}