#pragma once
#ifndef RRAM
#define RRAM
#include <cstdint>
#include <cstring>
#include <cmath>
#define NOISEMAX 1
#define LARGE 1000000

class RRAM_Array
{
public:
	// return the cycles it takes to complete this operation
	int load_weight(
		std::int8_t w_1[1024][1024],
		std::int8_t w_2[1024][1024],
		std::int8_t w_3[1024][1024],
		float bn_mean_1[1024],
		float bn_mean_2[1024],
		float bn_mean_3[1024],
		float bn_var_1[1024],
		float bn_var_2[1024],
		float bn_var_3[1024],
		float bn_weight_1[1024],
		float bn_weight_2[1024],
		float bn_weight_3[1024],
		float bn_bias_1[1024],
		float bn_bias_2[1024],
		float bn_bias_3[1024])
	{
		srand(time(NULL));
		for (int i = 0; i < 1024; i++)
		{
			for (int j = 0; j < 1024; j++)
			{
				weight_positive_1[i][j] = weight_negative_1[i][j] = 0;
				weight_positive_2[i][j] = weight_negative_2[i][j] = 0;
				weight_positive_3[i][j] = weight_negative_3[i][j] = 0;
				if (w_1[i][j] > 0)
				{
					weight_positive_1[i][j] = 1;
				}
				if (w_1[i][j] < 0)
				{
					weight_negative_1[i][j] = 1;
				}
				if (w_2[i][j] > 0)
				{
					weight_positive_2[i][j] = 1;
				}
				if (w_2[i][j] < 0)
				{
					weight_negative_2[i][j] = 1;
				}
				if (w_3[i][j] > 0)
				{
					weight_positive_3[i][j] = 1;
				}
				if (w_3[i][j] < 0)
				{
					weight_negative_3[i][j] = 1;
				}
			}
		}
		std::memcpy(batchnorm_mean_1, bn_mean_1, 4096);
		std::memcpy(batchnorm_var_1, bn_var_1, 4096);
		std::memcpy(batchnorm_weight_1, bn_weight_1, 4096);
		std::memcpy(batchnorm_bias_1, bn_bias_1, 4096);
		std::memcpy(batchnorm_mean_2, bn_mean_2, 4096);
		std::memcpy(batchnorm_var_2, bn_var_2, 4096);
		std::memcpy(batchnorm_weight_2, bn_weight_2, 4096);
		std::memcpy(batchnorm_bias_2, bn_bias_2, 4096);
		std::memcpy(batchnorm_mean_3, bn_mean_3, 4096);
		std::memcpy(batchnorm_var_3, bn_var_3, 4096);
		std::memcpy(batchnorm_weight_3, bn_weight_3, 4096);
		std::memcpy(batchnorm_bias_3, bn_bias_3, 4096);
		return 300000;
	}
	// return the cycles it takes to complete this operation
	int calculate(float input[1024], float output[1024], int fc)
	{
		int8_t(*weight_positive_)[1024];
		int8_t(*weight_negative_)[1024];
		float *batchnorm_mean_;
		float *batchnorm_var_;
		float *batchnorm_weight_;
		float *batchnorm_bias_;
		if (fc == 1)
		{
			weight_positive_ = weight_positive_1;
			weight_negative_ = weight_negative_1;
			batchnorm_mean_ = batchnorm_mean_1;
			batchnorm_var_ = batchnorm_var_1;
			batchnorm_weight_ = batchnorm_weight_1;
			batchnorm_bias_ = batchnorm_bias_1;
		}
		else if (fc == 2)
		{
			weight_positive_ = weight_positive_2;
			weight_negative_ = weight_negative_2;
			batchnorm_mean_ = batchnorm_mean_2;
			batchnorm_var_ = batchnorm_var_2;
			batchnorm_weight_ = batchnorm_weight_2;
			batchnorm_bias_ = batchnorm_bias_2;
		}
		else if (fc == 3)
		{
			weight_positive_ = weight_positive_3;
			weight_negative_ = weight_negative_3;
			batchnorm_mean_ = batchnorm_mean_3;
			batchnorm_var_ = batchnorm_var_3;
			batchnorm_weight_ = batchnorm_weight_3;
			batchnorm_bias_ = batchnorm_bias_3;
		}
		for (int i = 0; i < 1024; i++)
		{
			float value = 0;
			for (int j = 0; j < 1024; j++)
			{
				value += input[j] * weight_positive_[i][j];
			}
			for (int j = 0; j < 1024; j++)
			{
				value -= input[j] * weight_negative_[i][j];
			}
			output[i] = (value - batchnorm_mean_[i]) / std::sqrt(batchnorm_var_[i] + 1e-5f) * batchnorm_weight_[i] + batchnorm_bias_[i];
		}
		return 100;
	}
	// return the cycles it takes to complete this operation
	int sign(float input[1024], float output[1024])
	{
		float noise = ((rand() % LARGE) / LARGE - 0.5) * NOISEMAX;
		for (int i = 0; i < 1024; i++)
		{
			output[i] =
				input[i] > 0 ? (1 + noise) : input[i] < 0 ? (-1 + noise)
														  : 0;
		}
		return 1;
	}

private:
	int8_t weight_positive_1[1024][1024];
	int8_t weight_negative_1[1024][1024];
	float batchnorm_mean_1[1024];
	float batchnorm_var_1[1024];
	float batchnorm_weight_1[1024];
	float batchnorm_bias_1[1024];
	int8_t weight_positive_2[1024][1024];
	int8_t weight_negative_2[1024][1024];
	float batchnorm_mean_2[1024];
	float batchnorm_var_2[1024];
	float batchnorm_weight_2[1024];
	float batchnorm_bias_2[1024];
	int8_t weight_positive_3[1024][1024];
	int8_t weight_negative_3[1024][1024];
	float batchnorm_mean_3[1024];
	float batchnorm_var_3[1024];
	float batchnorm_weight_3[1024];
	float batchnorm_bias_3[1024];
};
#endif //RRAM