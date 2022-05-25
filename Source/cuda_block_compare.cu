#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void loop()
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x; //thread num 변수화
	printf("This is iteration number %d\n", idx);
}

int main()
{
	clock_t st = clock(); //구동시간 체크 시작
	int N = 10;

	loop << <1, N >> > (); //single block (168ms)
	loop << <2, N / 2 >> > (); //multi block (152ms) 

	cudaDeviceSynchronize(); //thread연산 동기화

	clock_t ed = clock(); //구동시간 체크 종료
	printf("time %u ms\n", ed - st); //구동시간 out
}