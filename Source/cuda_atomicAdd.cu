#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#define _CRT_SECURE_NO_WARNINGS


#define SIZE 1024
#define THREADS_PER_BLOCK 16

__global__ void VectorDot(int* a, int* b, int* c)
{
	__shared__ int temp[THREADS_PER_BLOCK]; //block sheard mem�� temp�� ����
	int t_id = threadIdx.x + blockIdx.x * blockDim.x;
	temp[threadIdx.x] = a[t_id] * b[t_id]; //�� thread�� a, b ������ ����
	__syncthreads(); //block�� thread�� ���� ����ȭ

	int sum = 0;
	if (threadIdx.x == 0)
	{
		for (int i = 0; i < THREADS_PER_BLOCK; i++) //block�� thread �������� �ջ�
		{
			sum += temp[i];
		}
		atomicAdd(c, sum);
	}
}

int main()
{
	int* a, * b, * c;
	int* d_a, * d_b, * d_c;

	a = (int*)malloc(SIZE * sizeof(int));  // malloc�� ���� host(cpu)������ ������� Ȯ��
	b = (int*)malloc(SIZE * sizeof(int));
	c = (int*)malloc(sizeof(int));  // ������� �ϳ��� ���(a,b �������� �� ����).

	cudaMalloc(&d_a, (SIZE * sizeof(int))); //cudaMalloc�� ���� device�� ��� ������ ������� Ȯ�� 
	cudaMalloc(&d_b, (SIZE * sizeof(int)));
	cudaMalloc(&d_c, sizeof(int)); // ������� �ϳ��� ���(a,b �������� �� ����).

	for (int i = 0; i < SIZE; ++i) //host ���� �ʱⰪ ����
	{
		a[i] = i;
		b[i] = i;
	}

	cudaMemcpy(d_a, a, (SIZE * sizeof(int)), cudaMemcpyHostToDevice); // malloc���� ������ host���� ������� ũ��, device������ ����
	cudaMemcpy(d_b, b, (SIZE * sizeof(int)), cudaMemcpyHostToDevice);
	cudaMemset(d_c, 0, sizeof(int));

	VectorDot << < 4, 16 >> > (d_a, d_b, d_c); //�����Լ� device����, <<< ����ϼ� , ��ϴ� ��뾲���� ��>>>
	cudaDeviceSynchronize();

	cudaMemcpy(c, d_c, sizeof(int), cudaMemcpyDeviceToHost); //device������ host������ ����

	printf("Final Sum : %d\n", *c);

	free(a); //host ���� ������� ����
	free(b);
	free(c);

	cudaFree(d_a); //device ���� ������� ����
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}