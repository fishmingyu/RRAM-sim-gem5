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
		std::string weightMemName,
		std::string bnBiasName,
		std::string bnMeanName,
		std::string bnVarName,
		std::string bnWeightName);
	int op_calculate(
		std::string inMemName,
		std::string outMemName);
	int op_sign(
		std::string inMemName,
		std::string outMemName);

	int outputIndex = -1;
	MemHub memhub;
	RRAM_Array Rarray;
};