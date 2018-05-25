#include <iostream>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
//#include <helper_cuda.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <omp.h>
#include <device_launch_parameters.h>

#define max(a, b) a > b ? a : b
#define min(a, b) a < b ? a : b

struct Edge{
    long long int x;
};

__global__ void initialize_parent(int* parent, int n){
    int bid = blockIdx.x;
    int id = bid*blockDim.x + threadIdx.x;
    if(id < n)
        parent[id] = id;    
    return;
}

__global__ void initialize_active_edges(bool* active_edges, int e){
    int bid = blockIdx.x;
    int id = bid*blockDim.x + threadIdx.x;
    if(id < e)
        active_edges[id] = true;
    return;
}

///*
__global__ void accumulate(Edge* edge_list, bool* cross_edges, int* indices, int e){
	int bid = blockIdx.x;
	int id = bid*blockDim.x + threadIdx.x;
    Edge temp;
    temp.x = 0;
	if(id < e)
		if(cross_edges[id])
			temp = edge_list[id];
    __syncthreads();
    if(temp.x)
        edge_list[indices[id]] = temp;
	return;	
}
//*/

__global__ void update_states(int* parent, int* vertex_state, int n){
    int bid = blockIdx.x;
    int id = bid*blockDim.x + threadIdx.x;
    if(id < n)
        vertex_state[id] = parent[id] == id ? 0 : 1;    
    return;
}

__global__ void hook_init(int* parent, Edge* edge_list, int e){
    int bid = blockIdx.x;
    int id = bid*blockDim.x + threadIdx.x;
    long long int x;
    int u, v, mx, mn;
    if(id < e){
        x = edge_list[id].x;
        v = (int) x & 0xFFFFFFFF;
        u = (int) (x >> 32);

        mx = max(u, v);
        mn = u + v - mx;
        parent[mx] = mn;
    }
    return;
}

__global__ void hook_even(int* parent, Edge* edge_list, int e, bool* flag, bool* active_edges){
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int id = bid*blockDim.x + tid;
    long long int x;
    int u, v, mx, mn, parent_u, parent_v;
    __shared__ bool block_flag;
    if(tid == 0)
        block_flag = false;
    __syncthreads();
    if(id < e)
        if(active_edges[id]){
            x = edge_list[id].x;
            v = (int) x & 0xFFFFFFFF;
            u = (int) (x >> 32);
            
            parent_u = parent[u];
            parent_v = parent[v];

            mx = max(parent_u, parent_v);
            mn = parent_u + parent_v - mx;
            
            if(parent_u == parent_v)
                active_edges[id] = false;
            else{
                parent[mn] = mx;
                block_flag = true;
            }
        }
    __syncthreads();

    if(tid == 0)
        if(block_flag)
            *flag = true;
    return;
}

__global__ void hook_odd(int* parent, Edge* edge_list, int e, bool* flag, bool* active_edges){
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int id = bid*blockDim.x + tid;
    long long int x;
    int u, v, mx, mn, parent_u, parent_v;
    __shared__ bool block_flag;
    if(tid == 0)
        block_flag = false;
    __syncthreads();
    if(id < e)
        if(active_edges[id]){
            x = edge_list[id].x;
            v = (int) x & 0xFFFFFFFF;
            u = (int) (x >> 32);
            
            parent_u = parent[u];
            parent_v = parent[v];

            mx = max(parent_u, parent_v);
            mn = parent_u + parent_v - mx;
            
            if(parent_u == parent_v)
                active_edges[id] = false;
            else{
                parent[mx] = mn;
                block_flag = true;
            }
        }
    __syncthreads();

    if(tid == 0)
        if(block_flag)
            *flag = true;
    return;
}

__global__ void pointer_jumping(int* parent, int n, bool* flag){
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int id = bid*blockDim.x + tid;
    int parent_id, grandparent_id;
    __shared__ bool block_flag;
    if(tid == 0)
        block_flag = false;
    __syncthreads();
    if(id < n){
        parent_id = parent[id];
        grandparent_id = parent[parent_id];
        if(parent_id != grandparent_id){
            parent[id] = grandparent_id;
            block_flag = true;
        }
    }
    if(tid == 0)
        if(block_flag)
            *flag = true;
    return;
}

__global__ void root_pointer_jumping(int* parent, int* vertex_state, int n, bool* flag){
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int id = bid*blockDim.x + tid;
    int parent_id, grandparent_id;
    __shared__ bool block_flag;
    if(tid == 0)
        block_flag = false;
    __syncthreads();
    if(id < n)
        if(vertex_state[id] == 0){
            parent_id = parent[id];
            grandparent_id = parent[parent_id];
            if(parent_id != grandparent_id){
                parent[id] = grandparent_id;
                block_flag = true;
            }
            else
                vertex_state[id] = -1;
        }
    if(tid == 0)
        if(block_flag)
            *flag = true;
    return;
}

__global__ void leaf_pointer_jumping(int* parent, int* vertex_state, int n){
    int bid = blockIdx.x;
    int id = bid*blockDim.x + threadIdx.x; 
    int parent_id, grandparent_id;
    if(id < n)
        if(vertex_state[id] == 1){
            parent_id = parent[id];
            grandparent_id = parent[parent_id];
            parent[id] = grandparent_id;
        }
    return;
}

__global__ void process_cross_edges(int* parent, Edge* edge_list, int e, bool* flag, bool* cross_edges){
	int bid = blockIdx.x;
    int tid = threadIdx.x;
    int id = bid*blockDim.x + tid;
    long long int x;
    int u, v, mn, mx, parent_u, parent_v;
    __shared__ bool block_flag;
    if(tid == 0)
        block_flag = false;
    __syncthreads();
    if(id < e)
        if(cross_edges[id]){
            x = edge_list[id].x;
            v = (int) x & 0xFFFFFFFF;
            u = (int) (x >> 32);
            
            parent_u = parent[u];
            parent_v = parent[v];

            mn = min(parent_u, parent_v);
			mx = parent_u + parent_v - mn;
            
            if(parent_u == parent_v)
                cross_edges[id] = false;
            else{
                parent[mx] = mn;
                block_flag = true;
            }
        }
    __syncthreads();

    if(tid == 0)
        if(block_flag)
            *flag = true;
    return;
} 


