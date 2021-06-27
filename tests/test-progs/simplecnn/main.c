#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/mman.h>
#include <stdint.h>
#include <assert.h>
#include "../../../util/m5/m5op.h"

#include "peripheral.h"
#include "rram_array.h"
#include "output_label.h"

#define BATCH 300
#define SIZE_FC_KERNEL (1024 * 1024)
#define SIZE_BN (1024)

#define TOTAL_NUM 300 // no more than 5135

int8_t fc1_w[1024][1024] = {0};
int8_t fc2_w[1024][1024] = {0};
int8_t fc3_w[1024][1024] = {0};

float bn1_b[1024] = {0};
float bn1_w[1024] = {0};
float bn1_var[1024] = {0};
float bn1_mean[1024] = {0};

float bn2_b[1024] = {0};
float bn2_w[1024] = {0};
float bn2_var[1024] = {0};
float bn2_mean[1024] = {0};

float bn3_b[1024] = {0};
float bn3_w[1024] = {0};
float bn3_var[1024] = {0};
float bn3_mean[1024] = {0};

void load_weights()
{
	// load weight data from files
	FILE *fdata = NULL;

	fdata = fopen("data/fc1_weight.txt", "r");
	assert(fdata != NULL);
	for (int i = 0; i < 64; i++)
	{
		for (int j = 0; j < 784; j++)
		{
			fscanf(fdata, "%hhd,", &fc1_w[i][j]);
		}
	}
	fclose(fdata);

	fdata = fopen("data/fc2_weight.txt", "r");
	assert(fdata != NULL);
	for (int i = 0; i < 256; i++)
	{
		for (int j = 0; j < 64; j++)
		{
			fscanf(fdata, "%hhd,", &fc2_w[i][j]);
		}
	}
	fclose(fdata);

	fdata = fopen("data/fc3_weight.txt", "r");
	assert(fdata != NULL);
	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < 256; j++)
		{
			fscanf(fdata, "%hhd,", &fc3_w[i][j]);
		}
	}
	fclose(fdata);

	fdata = fopen("data/bn1_bias.txt", "r");
	assert(fdata != NULL);
	for (int i = 0; i < 64; i++)
	{
		fscanf(fdata, "%f,", &bn1_b[i]);
	}
	fclose(fdata);

	fdata = fopen("data/bn1_weight.txt", "r");
	assert(fdata != NULL);
	for (int i = 0; i < 64; i++)
	{
		fscanf(fdata, "%f,", &bn1_w[i]);
	}
	fclose(fdata);

	fdata = fopen("data/bn1_var.txt", "r");
	assert(fdata != NULL);
	for (int i = 0; i < 64; i++)
	{
		fscanf(fdata, "%f,", &bn1_var[i]);
	}
	fclose(fdata);

	fdata = fopen("data/bn1_mean.txt", "r");
	assert(fdata != NULL);
	for (int i = 0; i < 64; i++)
	{
		fscanf(fdata, "%f,", &bn1_mean[i]);
	}
	fclose(fdata);

	fdata = fopen("data/bn2_bias.txt", "r");
	assert(fdata != NULL);
	for (int i = 0; i < 256; i++)
	{
		fscanf(fdata, "%f,", &bn2_b[i]);
	}
	fclose(fdata);

	fdata = fopen("data/bn2_weight.txt", "r");
	assert(fdata != NULL);
	for (int i = 0; i < 256; i++)
	{
		fscanf(fdata, "%f,", &bn2_w[i]);
	}
	fclose(fdata);

	fdata = fopen("data/bn2_var.txt", "r");
	assert(fdata != NULL);
	for (int i = 0; i < 256; i++)
	{
		fscanf(fdata, "%f,", &bn2_var[i]);
	}
	fclose(fdata);

	fdata = fopen("data/bn2_mean.txt", "r");
	assert(fdata != NULL);
	for (int i = 0; i < 256; i++)
	{
		fscanf(fdata, "%f,", &bn2_mean[i]);
	}
	fclose(fdata);

	fdata = fopen("data/bn3_bias.txt", "r");
	assert(fdata != NULL);
	for (int i = 0; i < 10; i++)
	{
		fscanf(fdata, "%f,", &bn3_b[i]);
	}
	fclose(fdata);

	fdata = fopen("data/bn3_weight.txt", "r");
	assert(fdata != NULL);
	for (int i = 0; i < 10; i++)
	{
		fscanf(fdata, "%f,", &bn3_w[i]);
	}
	fclose(fdata);

	fdata = fopen("data/bn3_var.txt", "r");
	assert(fdata != NULL);
	for (int i = 0; i < 10; i++)
	{
		fscanf(fdata, "%f,", &bn3_var[i]);
	}
	fclose(fdata);

	fdata = fopen("data/bn3_mean.txt", "r");
	assert(fdata != NULL);
	for (int i = 0; i < 10; i++)
	{
		fscanf(fdata, "%f,", &bn3_mean[i]);
	}
	fclose(fdata);
}

int main()
{
	/*****************************/
	m5_dumpreset_stats(0, 0);
	/*****************************/

	printf("Program Start.\n");

	volatile uint8_t *data = (uint8_t *)mmap(PERI_ADDR[0], 10 * 1024 * 1024, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
	printf("Peripherals Registered.\n");

	periInit(data, 0);
	printf("Inited.\n");

	load_weights();

	size_t vdev_offset = 2 + 1024 * 1024 * 2;

	periWrite(data, vdev_offset, (void *)fc1_w, SIZE_FC_KERNEL * sizeof(int8_t));
	vdev_offset += SIZE_FC_KERNEL * sizeof(int8_t);
	printf("fc1 Writen to Vdev.\n");

	periWrite(data, vdev_offset, bn1_b, SIZE_BN * sizeof(float));
	vdev_offset += SIZE_BN * sizeof(float);
	printf("bn1_b Writen to Vdev.\n");

	periWrite(data, vdev_offset, bn1_mean, SIZE_BN * sizeof(float));
	vdev_offset += SIZE_BN * sizeof(float);
	printf("bn1_mean Writen to Vdev.\n");

	periWrite(data, vdev_offset, bn1_var, SIZE_BN * sizeof(float));
	vdev_offset += SIZE_BN * sizeof(float);
	printf("bn1_var Writen to Vdev.\n");

	periWrite(data, vdev_offset, bn1_w, SIZE_BN * sizeof(float));
	vdev_offset += SIZE_BN * sizeof(float);
	printf("bn1_w Writen to Vdev.\n");

	periWrite(data, vdev_offset, fc2_w, SIZE_FC_KERNEL * sizeof(int8_t));
	vdev_offset += SIZE_FC_KERNEL * sizeof(int8_t);
	printf("fc2 Writen to Vdev.\n");

	periWrite(data, vdev_offset, bn2_b, SIZE_BN * sizeof(float));
	vdev_offset += SIZE_BN * sizeof(float);
	printf("bn2_b Writen to Vdev.\n");

	periWrite(data, vdev_offset, bn2_mean, SIZE_BN * sizeof(float));
	vdev_offset += SIZE_BN * sizeof(float);
	printf("bn2_mean Writen to Vdev.\n");

	periWrite(data, vdev_offset, bn2_var, SIZE_BN * sizeof(float));
	vdev_offset += SIZE_BN * sizeof(float);
	printf("bn2_var Writen to Vdev.\n");

	periWrite(data, vdev_offset, bn2_w, SIZE_BN * sizeof(float));
	vdev_offset += SIZE_BN * sizeof(float);
	printf("bn2_w Writen to Vdev.\n");

	periWrite(data, vdev_offset, fc3_w, SIZE_FC_KERNEL * sizeof(int8_t));
	vdev_offset += SIZE_FC_KERNEL * sizeof(int8_t);
	printf("fc3 Writen to Vdev.\n");

	periWrite(data, vdev_offset, bn3_b, SIZE_BN * sizeof(float));
	vdev_offset += SIZE_BN * sizeof(float);
	printf("bn3_b Writen to Vdev.\n");

	periWrite(data, vdev_offset, bn3_mean, SIZE_BN * sizeof(float));
	vdev_offset += SIZE_BN * sizeof(float);
	printf("bn3_mean Writen to Vdev.\n");

	periWrite(data, vdev_offset, bn3_var, SIZE_BN * sizeof(float));
	vdev_offset += SIZE_BN * sizeof(float);
	printf("bn3_var Writen to Vdev.\n");

	periWrite(data, vdev_offset, bn3_w, SIZE_BN * sizeof(float));
	vdev_offset += SIZE_BN * sizeof(float);
	printf("bn3_w Writen to Vdev.\n");

	FILE *fdata = fopen("data/images-mnist-test-01347", "rb");
	assert(fdata != NULL);

	/*****************************/
	m5_dumpreset_stats(0, 0);
	/*****************************/

	uint8_t finalresult[TOTAL_NUM];
	int count;
	uint8_t j;
	for (count = 0; count < BATCH; count++)
	{
		int8_t temp_int8[1024];
		float temp_float[1024];
		uint8_t input_byte[784];
		fread(input_byte, sizeof(input_byte), 1, fdata); //read input img
		for (int i = 0; i < 784; i++)
		{
			temp_int8[i] = input_byte[i] > 0 ? 1 : 0;
		}
		periWrite(data, 2 + 1024 * 1024, temp_int8, sizeof(temp_int8));

		for (j = 1; j <= 10; j++)
		{
			periInit(data, j);
		}
		assert(periIsFinished(data));
		periRead(data, 2 + 1024 * 1024, finalresult + count, sizeof(unsigned char));
		printf("Got result: %d\n", finalresult[count]);
	}
	fclose(fdata);
	periLogout(0);

	/*****************************/
	m5_dumpreset_stats(0, 0);
	/*****************************/

	uint8_t target_list[TOTAL_NUM];
	fdata = fopen("data/labels-mnist-test-01347", "rb"); // read ground truth
	assert(fdata != NULL);
	fread(target_list, sizeof(target_list), 1, fdata);
	int i;
	int correct = 0;
	for (i = 0; i < BATCH; i++)
	{
		if (finalresult[i] == target_list[i])
			correct++;
	}
	printf("accu: %f\n", (double)correct / (double)BATCH);
	munmap((void *)data, 1024 * 1024 * 10);
	return 0;
}
