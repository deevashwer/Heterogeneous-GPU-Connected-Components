#include <iostream>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <list>
#include <omp.h>
#include <vector>
#include "CC-kernel.cu"

#define THLD 55//40
#define SCALE 100
#define NUM_CPU_CORES 29//10//4
#define NUM_THREADS 512
#define PATH_TO_GRAPH "web-BerkStan.txt"
#define PATH_TO_RESULTS "Results-BerkStan.txt"

using namespace std;

struct MappedEdge{
    int v;
    int index;
};

int root(int* parent, int u, int offset){
    while(parent[u - offset] != u){
        parent[u - offset] = parent[parent[u - offset] - offset];
        u = parent[u - offset];
    }
    return u;
}

void hook(int* parent, int offset, int u, int v, int* size = NULL){
    int root_u = root(parent, u, offset);
    int root_v = root(parent, v, offset);
    if(root_u == root_v)
        return;
    if(size){
        if(size[root_u - offset] < size[root_v - offset]){
            parent[root_u - offset] = root_v;
            size[root_v - offset] += size[root_u - offset];
        }
        else{
            parent[root_v - offset] = root_u;
            size[root_u - offset] += size[root_v - offset];
        }
    }
    else{
        if(root_u < root_v)
            parent[root_u - offset] = root_v;
        else
            parent[root_v - offset] = root_u;
    }
    return;
}

void multilevel_pointer_jumping(int* visited, int n, int global_offset){
    //int* temp = new int[n];
    //omp_set_num_threads(NUM_CPU_CORES);
//#pragma omp parallel
    {
        int id = omp_get_thread_num();
        int nthrds = omp_get_num_threads() - 1;
        int offset = (id < (n % nthrds)) ? (n/nthrds)*id + id : (n/nthrds)*id + (n % nthrds);
        offset += global_offset;
        int N = (id < (n % nthrds)) ? (n/nthrds) + 1 : (n/nthrds);
        int *temp = new int[N];
        for(int i = 0; i < N; i++)
            temp[i] = root(visited, i + offset, 0);
        for(int i = 0; i < N; i++)
            visited[i + offset] = temp[i];
    }
}

void find_connected_components(int* visited, vector<MappedEdge>* edge_list, bool *cross_edges, int n, int global_offset){
    //omp_set_num_threads(NUM_CPU_CORES);
//#pragma omp parallel
    {
        int id = omp_get_thread_num();
        int nthrds = omp_get_num_threads() - 1;
        int offset = (id < (n % nthrds)) ? (n/nthrds)*id + id : (n/nthrds)*id + (n % nthrds);
        offset += global_offset;
        int N = (id < (n % nthrds)) ? (n/nthrds) + 1 : (n/nthrds);
        //int local_visited[N];
        //int size[N];
        //int *size = new int[N];
        /*
        for(int i = 0; i < N; i++){
            local_visited[i] = visited[i + offset];
            //size[i] = 1;
        }
        */
        int v, index;
        vector<MappedEdge>::iterator it;
        //#pragma omp for
        for(int i = 0; i < N; i++)
            for(it = edge_list[i + offset].begin(); it != edge_list[i + offset].end(); it++){
                v = (*it).v;
                index = (*it).index;
                if(v >= offset && v < offset + N){
                    //hook(local_visited, offset, i + offset, v);
                    hook(visited, 0, i + offset, v);
                    //cross_edges[index] = false;
                }
                else
                    cross_edges[index] = true;
            }
        /*
        for(int i = 0; i < N; i++)
            visited[i + offset] = local_visited[i];
        */
    }
    return;
}

///*
void process_cross_edges(int* parent, Edge* edge_list, bool* cross_edges, int e){
    omp_set_num_threads(NUM_CPU_CORES);
#pragma omp parallel for
    for(int i = 0; i < e; i++){
        if(cross_edges[i]){
            long long int x;
            int u, v;
            x = edge_list[i].x;
            v = (int) (x & 0xFFFFFFFF);
            u = (int) (x >> 32);
            hook(parent, 0, u, v);
        }
    }
    return;
}
//*/


/*
void identify_edges(Edge* edge_list, bool* cross_edges, int e, int n){
    for(int i = 0; i < e; i++){
        if(edge_list[i].u < n || edge_list[i].v < n){
            if(edge_list[i].u >= n || edge_list[i].v >= n){
                cross_edges[i] = true;
                //cout << edge_list[i].u << " " << edge_list[i].v << endl;
            }
            else
                ;//active_edges[i] = true;
        }
    }
    return;
}
*/

int main(int argc, const char * argv[]) {
    double start, end;
    //omp_set_num_threads(2);
    omp_set_num_threads(NUM_CPU_CORES + 1);
    omp_set_nested(1);
    cout << "Threshold: " << THLD << endl;
    cout << "Number of CPU Threads: " << NUM_CPU_CORES << endl;
    ///*
    ifstream input_file;
    input_file.open(PATH_TO_GRAPH);
    //input_file.seekg(9);
    string input;
    input_file >> input;
    int number_of_vertices = atoi(input.c_str());
    int number_of_vertices_CPU = ((double) THLD/SCALE) * number_of_vertices;
    int number_of_vertices_GPU = number_of_vertices - number_of_vertices_CPU;
    int global_offset = number_of_vertices_GPU;
    //number_of_vertices = 20;
    //input_file.seekg(8, ios::cur);
    input_file >> input;
    int number_of_edges = atoi(input.c_str());
    //input_file.seekg(23, ios::cur);
    //*/
    
    //list<Edge> edge_list;
    //list<int>* edge_list = new list<int>[number_of_vertices];
    
    ///*
    int u, v;
    Edge* edge_list = new Edge[number_of_edges];
    Edge* GPU_edge_list = new Edge[number_of_edges];
    Edge* cross_edge_list_CPU = new Edge[number_of_edges];
    //Edge* cross_edge_list = new Edge[number_of_edges];
    vector<MappedEdge>* edge_list_CPU = new vector<MappedEdge>[number_of_vertices];
    int number_of_edges_GPU = 0;//, cross_list_count = 0;
	Edge e;
	MappedEdge E;
    long long int x;
    bool *cross_edges;
    cross_edges = new bool[number_of_edges];
    for(int i = 0; i < number_of_edges; i++)
        cross_edges[i] = false;
    //int default_bucket_size = number_of_vertices_CPU/NUM_CPU_CORES;
    //int num_greater_buckets = number_of_vertices_CPU % NUM_CPU_CORES;
    //int bucket_number_u, bucket_number_v, temp;
    for(int i = 0; i < number_of_edges; i++){
        input_file >> input;
        u = atoi(input.c_str()) - 1;
        input_file >> input;
        v = atoi(input.c_str()) - 1;
        x = (long long int) u << 32;
        x += (long long int) v;
		e.x = x;
		E.v = v;
		E.index = i;
        edge_list[i] = e;
        if(u < global_offset){
            if(v < global_offset)
                GPU_edge_list[number_of_edges_GPU++] = e;
            else
                cross_edges[i] = true;
        }
        else if(v < global_offset){
            if(u < global_offset)
                GPU_edge_list[number_of_edges_GPU++] = e;
            else
                cross_edges[i] = true;
        }
        else edge_list_CPU[u].push_back(E);
    }
    //cout << "Time taken (Edge List Generation): " << (double)(CPU_total)/CLOCKS_PER_SEC << endl;
    //*/
    
    int *parent;
    parent = new int[number_of_vertices];
    
    int number_of_threads = NUM_THREADS;
    int number_of_blocks_N = (number_of_vertices)/NUM_THREADS + 1;
    //int number_of_blocks_E = (number_of_edges)/NUM_THREADS + 1;
    int number_of_blocks_n = (number_of_vertices_GPU)/NUM_THREADS + 1;
    int number_of_blocks_e = (number_of_edges_GPU)/NUM_THREADS + 1;
    dim3 grid_N (number_of_blocks_N, 1);
    //dim3 grid_E (number_of_blocks_E, 1);
    dim3 grid_n (number_of_blocks_n, 1);
    dim3 grid_e (number_of_blocks_e, 1);
    dim3 threads (number_of_threads, 1);
    
    Edge *edge_list_GPU, *cross_edge_list;
    int *parent_GPU, *vertex_state, *new_e;
    bool *active_edges, *cross_edges_GPU;
    bool *hooking_flag, *jumping_flag;
    
    /*checkCudaErrors*/(cudaMalloc(&edge_list_GPU, sizeof(Edge)*number_of_edges_GPU));
    cudaMalloc(&cross_edge_list, sizeof(Edge)*number_of_edges);
    /*checkCudaErrors*/(cudaMalloc(&parent_GPU, sizeof(int)*number_of_vertices));
    /*checkCudaErrors*/(cudaMalloc(&vertex_state, sizeof(int)*number_of_vertices));
    /*checkCudaErrors*/(cudaMalloc(&active_edges, sizeof(bool)*number_of_edges_GPU));
    /*checkCudaErrors*/(cudaMalloc(&cross_edges_GPU, sizeof(bool)*number_of_edges));
    /*checkCudaErrors*///(cudaMalloc(&indices, sizeof(int)*number_of_edges));
    /*checkCudaErrors*/(cudaMallocHost(&hooking_flag, sizeof(bool)));
    /*checkCudaErrors*/(cudaMallocHost(&jumping_flag, sizeof(bool)));
    (cudaMallocHost(&new_e, sizeof(int)));
    /*checkCudaErrors*/(cudaMemcpy(edge_list_GPU, GPU_edge_list, sizeof(Edge)*number_of_edges_GPU, cudaMemcpyHostToDevice));
    //cudaMemcpy(edge_list_complete, edge_list, sizeof(Edge)*number_of_edges, cudaMemcpyHostToDevice);
    //thrust::device_ptr<bool> cross_edges_ptr = thrust::device_pointer_cast(cross_edges_GPU);
    //thrust::device_ptr<int> indices_ptr = thrust::device_pointer_cast(indices);
    
    double GPU_start, GPU_stop, total_start, total_stop, time_phase_1, time_phase_2;
    for(int i = number_of_vertices_GPU; i < number_of_vertices; i++)
        parent[i] = i;
	total_start = omp_get_wtime();
#pragma omp parallel
    {
        ///*
        int id = omp_get_thread_num();
        if(id < NUM_CPU_CORES){
        	#pragma omp master
        	{
        		start = omp_get_wtime();
        	}
            /*
             graph.find_connected_components(parent, cross_edges, global_offset);
             for(int i = global_offset; i < global_offset + number_of_vertices_CPU; i++)
             if(parent[i] == -1)
             parent[i] = i;
             */
            if(THLD > 0){
                find_connected_components(parent, edge_list_CPU, cross_edges, number_of_vertices_CPU, global_offset);
                //end = omp_get_wtime();
                //start = omp_get_wtime();
                multilevel_pointer_jumping(parent, number_of_vertices_CPU, global_offset);
            }
        	end = omp_get_wtime();
        }else if(THLD < SCALE && id == NUM_CPU_CORES){
            GPU_start = omp_get_wtime();
            
            initialize_parent<<<grid_n, threads>>>(parent_GPU, number_of_vertices_GPU);
            initialize_active_edges<<<grid_e, threads>>>(active_edges, number_of_edges_GPU);
            //cudaDeviceSynchronize();
            
            hook_init<<<grid_e, threads>>>(parent_GPU, edge_list_GPU, number_of_edges_GPU);
            //cudaDeviceSynchronize();
            
            do{
                *jumping_flag = false;
                pointer_jumping<<<grid_n, threads>>>(parent_GPU, number_of_vertices_GPU, jumping_flag);
                cudaDeviceSynchronize();
            }while(*jumping_flag);
            
            int iteration_flag = 0;
            do{
                *hooking_flag = false;
                update_states<<<grid_n, threads>>>(parent_GPU, vertex_state, number_of_vertices_GPU);
                
                if(iteration_flag % 2)
                    hook_odd<<<grid_e, threads>>>(parent_GPU, edge_list_GPU, number_of_edges_GPU, hooking_flag, active_edges);
                else
                    hook_even<<<grid_e, threads>>>(parent_GPU, edge_list_GPU, number_of_edges_GPU, hooking_flag, active_edges);
                cudaDeviceSynchronize();
                
                if(*hooking_flag == false)
                    break;
                
                do{
                    *jumping_flag = false;
                    root_pointer_jumping<<<grid_n, threads>>>(parent_GPU, vertex_state, number_of_vertices_GPU, jumping_flag);
                    cudaDeviceSynchronize();
                }while(*jumping_flag);
                
                leaf_pointer_jumping<<<grid_n, threads>>>(parent_GPU, vertex_state, number_of_vertices_GPU);
                cudaDeviceSynchronize();
            }while(*hooking_flag);
            //cudaMemcpyAsync(edge_list_GPU, edge_list, sizeof(Edge)*number_of_edges, cudaMemcpyHostToDevice);
            //cudaDeviceSynchronize();
            ///*checkCudaErrors*/(cudaMemcpy(parent, parent_GPU, sizeof(int)*number_of_vertices_GPU, cudaMemcpyDeviceToHost));
            GPU_stop = omp_get_wtime();
        }
    }
	total_stop = omp_get_wtime();
	cudaDeviceSynchronize();
    
    cout << "Time Taken (DSU Algorithm CPU): " << (end - start) << "s" << endl;
    cout << "Time Taken (SV Algorithm GPU): " << (GPU_stop - GPU_start) << "s" << endl;
	cout << "Time Taken (Phase I): " << (total_stop - total_start) << "s" << endl;
	double total_time = total_stop - total_start;
	time_phase_1 = total_time;
    //(cudaMemcpy(parent, parent_GPU, sizeof(int)*number_of_vertices_GPU, cudaMemcpyDeviceToHost));
	total_start = omp_get_wtime();
	int count = 0;
	for(int i = 0; i < number_of_edges; i++){
		if(cross_edges[i])
			cross_edge_list_CPU[count++] = edge_list[i];
		cross_edges[i] = true;
	}
	number_of_edges = count;
	cout << "Number of cross edges: " << number_of_edges << endl;
	cudaMemcpy(cross_edge_list, cross_edge_list_CPU, sizeof(Edge)*number_of_edges, cudaMemcpyHostToDevice);
	cudaMemcpy(cross_edges_GPU, cross_edges, sizeof(bool)*number_of_edges, cudaMemcpyHostToDevice);
	cudaMemcpy(parent_GPU + number_of_vertices_GPU, parent + number_of_vertices_GPU, sizeof(int)*number_of_vertices_CPU, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	total_stop = omp_get_wtime();
	int number_of_blocks_E = (number_of_edges)/NUM_THREADS + 1;
    dim3 grid_E (number_of_blocks_E, 1);


    total_start = omp_get_wtime();
    if(THLD > 0){
        if(THLD != SCALE || NUM_CPU_CORES > 1){
        	/*
        	thrust::inclusive_scan(cross_edges_ptr, cross_edges_ptr + number_of_edges, indices_ptr);
        	cudaMemcpy(new_e, indices + number_of_edges - 1, sizeof(int), cudaMemcpyDeviceToHost);
            accumulate<<<grid_E, threads>>>(edge_list_complete, cross_edges_GPU, indices, number_of_edges);
            number_of_edges = 454599;//*new_e;
            initialize_active_edges<<<(number_of_edges/NUM_THREADS), threads>>>(cross_edges_GPU, number_of_edges);
            cout << *new_e << endl;
			//*/
        	 do{
                *hooking_flag = false;
                update_states<<<grid_N, threads>>>(parent_GPU, vertex_state, number_of_vertices);
                
                process_cross_edges<<<grid_E, threads>>>(parent_GPU, cross_edge_list, number_of_edges, hooking_flag, cross_edges_GPU);
                cudaDeviceSynchronize();
                
                if(*hooking_flag == false)
                    break;
                ///*
                do{
                    *jumping_flag = false;
                    root_pointer_jumping<<<grid_N, threads>>>(parent_GPU, vertex_state, number_of_vertices, jumping_flag);
                    cudaDeviceSynchronize();
                }while(*jumping_flag);
                
                leaf_pointer_jumping<<<grid_N, threads>>>(parent_GPU, vertex_state, number_of_vertices);
                cudaDeviceSynchronize();
                //*/
            }while(*hooking_flag);
            //*/
           //process_cross_edges(parent, edge_list, cross_edges, number_of_edges);
           //multilevel_pointer_jumping(parent, number_of_vertices, 0);
        }
    }
    total_stop = omp_get_wtime();
    cout << "Time Taken (Cross Edges Processsing - Phase II): " << (total_stop - total_start) << "s" << endl;
    total_time += (total_stop - total_start);
    time_phase_2 = (total_stop - total_start);
    cout << "Time Taken (Total): " << total_time << "s" << endl;

    //(cudaMemcpy(parent, parent_GPU, sizeof(int)*number_of_vertices, cudaMemcpyDeviceToHost));
    if(THLD < 100 || NUM_CPU_CORES > 1)
    	(cudaMemcpy(parent, parent_GPU, sizeof(int)*number_of_vertices, cudaMemcpyDeviceToHost));
    
    //double temp_end = omp_get_wtime();
    
    //cout << "Time Taken (Cross Edge Handling - GPU): " << (double) GPU_time/1000000000 << "s" << endl;
    //cout << "Total Time: " << total_time << "s" << endl;
    //cout << "Total Time: " << temp_end - temp_start << "s" << endl;
    
    ///*
    cout << "Number of Elements in largest component: ";
    int temp = parent[0];
    count = 0;
    for(int i = 0; i < number_of_vertices; i++)
        if(parent[i] == temp)
            count++;
    //cout << i << "->" << parent[i] << endl;
    //cout << "(" << edge_list[i].u << ", " << edge_list[i].v << ")" << endl;
    cout << count << endl;
    //*/
    
    /*checkCudaErrors*/(cudaFree(parent_GPU));
    /*checkCudaErrors*/(cudaFree(edge_list_GPU));
    /*checkCudaErrors*/(cudaFree(vertex_state));
    /*checkCudaErrors*/(cudaFree(active_edges));

    ofstream output_file;
    output_file.open(PATH_TO_RESULTS, ofstream::out | ofstream::app);
    output_file << THLD << "\t" << NUM_CPU_CORES << "\t" << total_time << "\t" << time_phase_1 << "\t" << time_phase_2 << endl;
    output_file.close();

    return 0;
}
