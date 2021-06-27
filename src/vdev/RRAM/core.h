#pragma once
#include <fstream>
#include <string>

#include "memhub.h"
#include "tools.h"
#include "rram_array.h"

class Core
{
public:
	//the return values of operation functions below represent
	//how many cycles it takes
	int op_init();
	int op_input(std::string fromMemName, std::string toMemName);
	int op_output(std::string memName);
	int op_load(
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
		std::string bnWeight3Name);
	int op_calculate(
		std::string inMemName,
		std::string outMemName,
		int fc);
	int op_sign(
		std::string inMemName,
		std::string outMemName);

	int outputIndex = -1;
	MemHub memhub;
	RRAM_Array Rarray;
};