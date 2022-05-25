#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "device_launch_parameters.h"
#define SIZE 1024
#define THREADS_PER_BLOCK 16

__global__ void VectorAdd(int* a, int* b, int* c, int n)  //global ����� ����  device(gpu)�� �Լ� ����
{
	for (int i = 0; i < n; ++i)
		c[i] = a[i] + b[i];
}

int main()
{
	int* a, * b, * c;
	int* d_a, * d_b, * d_c;

	a = (int*)malloc(SIZE * sizeof(int));  // malloc�� ���� host(cpu)������ ������� Ȯ��
	b = (int*)malloc(SIZE * sizeof(int));
	c = (int*)malloc(SIZE * sizeof(int));

	cudaMalloc(&d_a, (SIZE * sizeof(int))); //cudaMalloc�� ���� device�� ��� ������ ������� Ȯ�� 
	cudaMalloc(&d_b, (SIZE * sizeof(int)));
	cudaMalloc(&d_c, (SIZE * sizeof(int)));

	for (int i = 0; i < SIZE; ++i) //host ���� �ʱⰪ ����
	{
		a[i] = i;
		b[i] = i;
		c[i] = 0;
	}

	cudaMemcpy(d_a, a, (SIZE * sizeof(int)), cudaMemcpyHostToDevice); // malloc���� ������ host���� ������� ũ��, device������ ����
	cudaMemcpy(d_b, b, (SIZE * sizeof(int)), cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, c, (SIZE * sizeof(int)), cudaMemcpyHostToDevice);


	VectorAdd << < SIZE / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (d_a, d_b, d_c, SIZE); //�����Լ� device����, <<< ����ϼ� , ��ϴ� ��뾲���� ��>>>

	cudaMemcpy(a, d_a, (SIZE * sizeof(int)), cudaMemcpyDeviceToHost);  //device���갪 host�� ����
	cudaMemcpy(b, d_b, (SIZE * sizeof(int)), cudaMemcpyDeviceToHost);
	cudaMemcpy(c, d_c, (SIZE * sizeof(int)), cudaMemcpyDeviceToHost);

	for (int i = 0; i < 10; ++i) //10ȸ ���� ����
		printf("c[%d] = %d\n", i, c[i]);

	free(a); //host ���� ������� ����
	free(b);
	free(c);

	cudaFree(d_a); //device ���� ������� ����
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}