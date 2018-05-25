#include <iostream>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <list>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <omp.h>
#include <vector>

#define THLD 0//40
#define NUM_CPU_CORES 4//4
#define NUM_THREADS 8192//32768//16384//8192//65536//128
#define WORK_GROUP_SIZE 256//64
#define PATH_TO_KERNEL "GPU-kernels.cl"
#define PATH_TO_GRAPH "web-Stanford.txt"

using namespace std;

struct Edge{
    int u, v;
};

struct Auxiliary{
    int num_threads;
    int group_size;
    bool flag;
    bool iter_flag;
    bool pointer_jumping_flag;
};

struct MappedEdge{
    int v;
    int index;
};

int root(int* visited, int u, int offset){
    while(visited[u - offset] != u){
        //visited[u - offset] = visited[visited[u - offset] - offset];
        u = visited[u - offset];
    }
    return u;
}

void hook(int* visited, int offset, int u, int v, int* size){
    int root_u = root(visited, u, offset);
    int root_v = root(visited, v, offset);
    if(root_u == root_v)
        return;
    //if(size[root_u - offset] < size[root_v - offset]){
    //if(root_u > root_v){
    if(root_u < root_v){
        visited[root_u - offset] = root_v;
        //size[root_v - offset] += size[root_u - offset];
    }
    else{
        visited[root_v - offset] = root_u;
        //size[root_u - offset] += size[root_v - offset];
    }
    return;
}

void multilevel_pointer_jumping(int* visited, int n, int global_offset){
    //int* temp = new int[n];
    omp_set_num_threads(NUM_CPU_CORES - 1);
#pragma omp parallel
    {
        int id = omp_get_thread_num();
        int nthrds = omp_get_num_threads();
        int offset = (id < (n % nthrds)) ? (n/nthrds)*id + id : (n/nthrds)*id + (n % nthrds);
        offset += global_offset;
        int N = (id < (n % nthrds)) ? (n/nthrds) + 1 : (n/nthrds);
        int temp[N];
        for(int i = 0; i < N; i++)
            temp[i] = root(visited, i + offset, 0);
        for(int i = 0; i < N; i++)
            visited[i + offset] = temp[i];
    }
}

void find_connected_components(int* visited, vector<MappedEdge>* edge_list, bool *cross_edges, int n, int global_offset){
    omp_set_num_threads(NUM_CPU_CORES);
#pragma omp parallel
    {
        int id = omp_get_thread_num();
        int nthrds = omp_get_num_threads();
        int offset = (id < (n % nthrds)) ? (n/nthrds)*id + id : (n/nthrds)*id + (n % nthrds);
        offset += global_offset;
        int N = (id < (n % nthrds)) ? (n/nthrds) + 1 : (n/nthrds);
        int local_visited[N];
        int size[N];
        //int *size = new int[N];
        for(int i = 0; i < N; i++){
            local_visited[i] = visited[i + offset];
            size[i] = 1;
        }
        int v, index;
        vector<MappedEdge>::iterator it;
        //#pragma omp for
        for(int i = 0; i < N; i++)
            for(it = edge_list[i + offset].begin(); it != edge_list[i + offset].end(); it++){
                v = (*it).v;
                index = (*it).index;
                if(v >= offset && v < offset + N){
                    hook(local_visited, offset, i + offset, v, size);
                    //cross_edges[index] = false;
                }
                else
                    cross_edges[index] = true;
            }
        for(int i = 0; i < N; i++)
            visited[i + offset] = local_visited[i];
    }
    return;
}

void process_cross_edges(int* visited, Edge* edge_list, bool* cross_edges, int e){
    omp_set_num_threads(NUM_CPU_CORES);
#pragma omp parallel for
    for(int i = 0; i < e; i++)
        if(cross_edges[i])
            hook(visited, 0, edge_list[i].u, edge_list[i].v, NULL);
    return;
}

int accumulate_edges(Edge* edge_list, int* offset, bool* active_edges, int e){
    int count = 0;
    for(int i = 0; i < e; i++){
        //cout << active_edges[i] << " " << i << " " << count << endl;
        if(active_edges[offset[i]])
            offset[count++] = offset[i];
    }
    return count;
}

int accumulate_vertices(int* node_type, int* offset, int n, int marker){
    int count = 0;
    for(int i = 0; i < n; i++)
        if(node_type[offset[i]] == marker)
            offset[count++] = offset[i];
    return count;
}

void identify_edges(Edge* edge_list, bool* active_edges, bool* cross_edges, int e, int n){
    for(int i = 0; i < e; i++){
        if(edge_list[i].u < n || edge_list[i].v < n){
            if(edge_list[i].u >= n || edge_list[i].v >= n){
                cross_edges[i] = true;
                //cout << edge_list[i].u << " " << edge_list[i].v << endl;
            }
            else
                active_edges[i] = true;
        }
    }
    return;
}

void heterogeneous_connected_components(){
    return;
}

char* read_kernel_from_file(char* path_to_file){
    FILE *fp;
    char *source_str;
    size_t source_size;
    fp = fopen(PATH_TO_KERNEL, "r");
    if (!fp) {
        printf("Failed to load kernel\n");
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    source_size = ftell(fp);
    rewind(fp);
    source_str = (char*) malloc(source_size + 1);
    source_str[source_size] = '\0';
    fread(source_str, sizeof(char), source_size, fp);
    fclose(fp);
    return source_str;
}

int main(int argc, const char * argv[]) {
    clock_t CPU_start, CPU_stop;
    double start, end, start_GPU, end_GPU, total_time;
    omp_set_num_threads(2);
    omp_set_nested(1);
    
    ///*
    ifstream input_file;
    input_file.open(PATH_TO_GRAPH);
    input_file.seekg(9);
    string input;
    input_file >> input;
    int number_of_vertices = atoi(input.c_str());
    int number_of_vertices_CPU = (THLD * number_of_vertices)/100;
    int number_of_vertices_GPU = number_of_vertices - number_of_vertices_CPU;
    int global_offset = number_of_vertices_GPU;
    //number_of_vertices = 20;
    input_file.seekg(8, ios::cur);
    input_file >> input;
    int number_of_edges = atoi(input.c_str());
    input_file.seekg(23, ios::cur);
    //*/
    
    //list<Edge> edge_list;
    //list<int>* edge_list = new list<int>[number_of_vertices];
    
    ///*
    int u, v;
    Edge* edge_list;// = new Edge[number_of_edges];
    if(posix_memalign((void **)&edge_list, 4096, sizeof(Edge)*number_of_edges) != 0)
        cout << "Error Allocating Memory" << endl;
    vector<MappedEdge>* edge_list_CPU = new vector<MappedEdge>[number_of_vertices];
    CPU_start = clock();
    for(int i = 0; i < number_of_edges; i++){
        input_file >> input;
        u = atoi(input.c_str()) - 1;
        input_file >> input;
        v = atoi(input.c_str()) - 1;
        edge_list[i] = {u, v};
        edge_list_CPU[u].push_back({v, i});
    }
    CPU_stop = clock();
    cout << "Time taken (Edge List Generation): " << (double)(CPU_stop - CPU_start)/CLOCKS_PER_SEC << endl;
    //*/
    
    /*
     list<Edge>::iterator it;
     for(it = edge_list.begin(); it != edge_list.end(); it++)
     cout << "(" <<(*it).u << ", " << (*it).v << ")" << endl;
     */
    
    Auxiliary auxiliary;
    int *visited, *offset, *offset_vertex, *node_type;
    bool *cross_edges, *active_edges;
    if(posix_memalign((void **)&auxiliary, 4096, sizeof(Auxiliary)) != 0)
        cout << "Error Allocating Memory" << endl;
    auxiliary.num_threads = NUM_THREADS;
    auxiliary.group_size = WORK_GROUP_SIZE;
    auxiliary.flag = false;
    auxiliary.iter_flag = true;
    if(posix_memalign((void **)&visited, 4096, number_of_vertices*sizeof(int)) != 0)
        cout << "Error Allocating Memory" << endl;
    if(posix_memalign((void **)&node_type, 4096, number_of_vertices*sizeof(int)) != 0)
        cout << "Error Allocating Memory" << endl;
    if(posix_memalign((void **)&offset, 4096, number_of_edges*sizeof(int)) != 0)
        cout << "Error Allocating Memory" << endl;
    if(posix_memalign((void **)&offset_vertex, 4096, number_of_vertices*sizeof(int)) != 0)
        cout << "Error Allocating Memory" << endl;
    if(posix_memalign((void **)&cross_edges, 4096, number_of_edges*sizeof(bool)) != 0)
        cout << "Error Allocating Memory" << endl;
    if(posix_memalign((void **)&active_edges, 4096, number_of_edges*sizeof(bool)) != 0)
        cout << "Error Allocating Memory" << endl;
    for(int i = 0; i < number_of_edges; i++){
        active_edges[i] = false;
        cross_edges[i] = false;
    }
    identify_edges(edge_list, active_edges, cross_edges, number_of_edges, number_of_vertices_GPU);
    //cout << accumulate(edge_list, cross_edges, number_of_edges) << endl;
    
    //CPU_start = clock();
    for(int i = 0; i < number_of_vertices; i++)
        visited[i] = i, offset_vertex[i] = i;
    for(int i = 0; i < number_of_edges; i++)
        offset[i] = i;
    //CPU_stop = clock();
    //cout << "Time taken (Array Initialization-CPU): " << (double)(CPU_stop - CPU_start)/CLOCKS_PER_SEC << "s" << endl;
    
    cl_device_id device;
    clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, NULL);
    cl_event event;
    cl_ulong GPU_start, GPU_stop, GPU_time = 0, GPU_jumping_time = 0, local_GPU_time = 0;
    
    char* source_str = read_kernel_from_file(PATH_TO_KERNEL);
    if(!source_str)
        return 1;
    
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)(&source_str), NULL, NULL);
    clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    //cl_kernel initialization_kernel = clCreateKernel(program, "initialize", NULL);
    cl_kernel hooking_kernel = clCreateKernel(program, "hook", NULL);
    cl_kernel pointer_jumping_kernel = clCreateKernel(program, "multi_pointer_jumping", NULL);
    cl_kernel pointer_jumping_masked_kernel = clCreateKernel(program, "multi_pointer_jumping_masked", NULL);
    cl_kernel pointer_jumping_unmasked_kernel = clCreateKernel(program, "multi_pointer_jumping_unmasked", NULL);
    cl_kernel update_mask_kernel = clCreateKernel(program, "update_mask", NULL);
    //cl_kernel process_cross_edges_kernel = clCreateKernel(program, "process_cross_edges", NULL);
    
    cl_mem auxiliary_GPU = clCreateBuffer(context, CL_MEM_USE_HOST_PTR, sizeof(Auxiliary), &auxiliary, NULL);
    cl_mem visited_GPU = clCreateBuffer(context, CL_MEM_USE_HOST_PTR, number_of_vertices*sizeof(int), visited, NULL);
    cl_mem edge_list_GPU = clCreateBuffer(context, CL_MEM_USE_HOST_PTR, number_of_edges*sizeof(Edge), edge_list, NULL);
    //cl_mem edge_list_GPU = clCreateBuffer(context, CL_MEM_READ_ONLY, number_of_edges*sizeof(Edge), NULL, NULL);
    //clEnqueueWriteBuffer(queue, edge_list_GPU, CL_TRUE, 0, number_of_edges*sizeof(Edge), edge_list, 0, NULL, NULL);
    cl_mem offset_GPU = clCreateBuffer(context, CL_MEM_USE_HOST_PTR, number_of_edges*sizeof(int), offset, NULL);
    cl_mem offset_vertex_GPU = clCreateBuffer(context, CL_MEM_USE_HOST_PTR, number_of_vertices*sizeof(int), offset_vertex, NULL);
    //cl_mem cross_edges_GPU = clCreateBuffer(context, CL_MEM_USE_HOST_PTR, number_of_edges*sizeof(bool), cross_edges, NULL);
    cl_mem active_edges_GPU = clCreateBuffer(context, CL_MEM_USE_HOST_PTR, number_of_edges*sizeof(bool), active_edges, NULL);
    cl_mem node_type_GPU = clCreateBuffer(context, CL_MEM_USE_HOST_PTR, number_of_vertices*sizeof(int), node_type, NULL);
    
    size_t global_dimensions[] = {NUM_THREADS, 0, 0};
    size_t local_dimensions[] = {WORK_GROUP_SIZE, 0, 0};
    //size_t *local_dimensions = NULL;
    
    /*
     clSetKernelArg(initialization_kernel, 0, sizeof(visited_GPU), &visited_GPU);
     clSetKernelArg(initialization_kernel, 1, sizeof(cl_int), &number_of_vertices);
     clSetKernelArg(initialization_kernel, 2, sizeof(auxiliary_GPU), &auxiliary_GPU);
     clEnqueueNDRangeKernel(queue, initialization_kernel, 1, NULL, global_dimensions, local_dimensions, 0, NULL, &event);
     clFinish(queue);
     clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &GPU_start, NULL);
     clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &GPU_stop, NULL);
     cout << "Time Taken (GPU): " << (double)(GPU_stop - GPU_start)/(1000000000) << endl;
     */
    
    int e = number_of_edges;
    int GPU_offset = 0;
    
    clSetKernelArg(hooking_kernel, 0, sizeof(visited_GPU), &visited_GPU);
    clSetKernelArg(hooking_kernel, 1, sizeof(edge_list_GPU), &edge_list_GPU);
    clSetKernelArg(hooking_kernel, 2, sizeof(offset_GPU), &offset_GPU);
    clSetKernelArg(hooking_kernel, 3, sizeof(active_edges_GPU), &active_edges_GPU);
    clSetKernelArg(hooking_kernel, 4, sizeof(cl_int), &e);
    clSetKernelArg(hooking_kernel, 5, sizeof(auxiliary_GPU), &auxiliary_GPU);
    
    clSetKernelArg(pointer_jumping_kernel, 0, sizeof(visited_GPU), &visited_GPU);
    clSetKernelArg(pointer_jumping_kernel, 1, sizeof(cl_int), &number_of_vertices_GPU);
    clSetKernelArg(pointer_jumping_kernel, 2, sizeof(auxiliary_GPU), &auxiliary_GPU);
    clSetKernelArg(pointer_jumping_kernel, 3, sizeof(cl_int), &GPU_offset);
    
    clSetKernelArg(pointer_jumping_masked_kernel, 0, sizeof(visited_GPU), &visited_GPU);
    clSetKernelArg(pointer_jumping_masked_kernel, 1, sizeof(node_type_GPU), &node_type_GPU);
    clSetKernelArg(pointer_jumping_masked_kernel, 2, sizeof(offset_vertex_GPU), &offset_vertex_GPU);
    clSetKernelArg(pointer_jumping_masked_kernel, 3, sizeof(cl_int), &number_of_vertices_GPU);
    clSetKernelArg(pointer_jumping_masked_kernel, 4, sizeof(auxiliary_GPU), &auxiliary_GPU);
    
    clSetKernelArg(pointer_jumping_unmasked_kernel, 0, sizeof(visited_GPU), &visited_GPU);
    clSetKernelArg(pointer_jumping_unmasked_kernel, 1, sizeof(node_type_GPU), &node_type_GPU);
    clSetKernelArg(pointer_jumping_unmasked_kernel, 2, sizeof(offset_vertex_GPU), &offset_vertex_GPU);
    clSetKernelArg(pointer_jumping_unmasked_kernel, 3, sizeof(cl_int), &number_of_vertices_GPU);
    clSetKernelArg(pointer_jumping_unmasked_kernel, 4, sizeof(auxiliary_GPU), &auxiliary_GPU);
    
    clSetKernelArg(update_mask_kernel, 0, sizeof(visited_GPU), &visited_GPU);
    clSetKernelArg(update_mask_kernel, 1, sizeof(node_type_GPU), &node_type_GPU);
    clSetKernelArg(update_mask_kernel, 2, sizeof(cl_int), &number_of_vertices_GPU);
    clSetKernelArg(update_mask_kernel, 3, sizeof(auxiliary_GPU), &auxiliary_GPU);
    clSetKernelArg(update_mask_kernel, 4, sizeof(cl_int), &GPU_offset);
    
    double temp_start = omp_get_wtime();
#pragma omp parallel
    {
        ///*
        int id = omp_get_thread_num();
        if(id == 0){
            start = omp_get_wtime();
            /*
             graph.find_connected_components(visited, cross_edges, global_offset);
             for(int i = global_offset; i < global_offset + number_of_vertices_CPU; i++)
             if(visited[i] == -1)
             visited[i] = i;
             */
            find_connected_components(visited, edge_list_CPU, cross_edges, number_of_vertices_CPU, global_offset);
            //end = omp_get_wtime();
            //start = omp_get_wtime();
            multilevel_pointer_jumping(visited, number_of_vertices_CPU, global_offset);
            end = omp_get_wtime();
        }else{
            ///*
            
            int i = 1, j = 1;
            e = number_of_edges;
            int n = number_of_vertices_GPU;
            while(auxiliary.flag == false){
                //while(i == 1){
                local_GPU_time = 0;
                ///*
                clEnqueueNDRangeKernel(queue, update_mask_kernel, 1, NULL, global_dimensions, local_dimensions, 0, NULL, &event);
                clFinish(queue);
                clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &GPU_start, NULL);
                clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &GPU_stop, NULL);
                GPU_time += (GPU_stop - GPU_start);
                GPU_jumping_time += (GPU_stop - GPU_start);
                local_GPU_time += (GPU_stop - GPU_start);
                //*/
                 
                auxiliary.flag = true;
                ///*
                start_GPU = omp_get_wtime();
                e = accumulate_edges(edge_list, offset, active_edges, e);
                end_GPU = omp_get_wtime();
                GPU_time += (end_GPU - start_GPU) * 1000000000;
                local_GPU_time += (end_GPU - start_GPU) * 1000000000;
                clSetKernelArg(hooking_kernel, 4, sizeof(cl_int), &e);
                //cout << number_of_edges << endl;
                //*/
                
                ///*
                start_GPU = omp_get_wtime();
                n = accumulate_vertices(node_type, offset_vertex, n, 0);
                end_GPU = omp_get_wtime();
                GPU_time += (end_GPU - start_GPU) * 1000000000;
                local_GPU_time += (end_GPU - start_GPU) * 1000000000;
                clSetKernelArg(pointer_jumping_masked_kernel, 4, sizeof(cl_int), &n);
                //cout << n << endl;
                //*/
                
                clEnqueueNDRangeKernel(queue, hooking_kernel, 1, NULL, global_dimensions, local_dimensions, 0, NULL, &event);
                clFinish(queue);
                clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &GPU_start, NULL);
                clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &GPU_stop, NULL);
                GPU_time += (GPU_stop - GPU_start);
                local_GPU_time += (GPU_stop - GPU_start);
                //multilevel_pointer_jumping(visited, number_of_vertices_GPU, 0);
                
                auxiliary.pointer_jumping_flag = true;
                cout << "Pointer Jumping Iteration...";
                j = 1;
                while(auxiliary.pointer_jumping_flag){
                    auxiliary.pointer_jumping_flag = false;
                    clEnqueueNDRangeKernel(queue, pointer_jumping_masked_kernel, 1, NULL, global_dimensions, local_dimensions, 0, NULL, &event);
                    clFinish(queue);
                    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &GPU_start, NULL);
                    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &GPU_stop, NULL);
                    GPU_time += (GPU_stop - GPU_start);
                    GPU_jumping_time += (GPU_stop - GPU_start);
                    local_GPU_time += (GPU_stop - GPU_start);
                    cout << j++ << "...";
                }
                cout << endl;
                
                clEnqueueNDRangeKernel(queue, pointer_jumping_unmasked_kernel, 1, NULL, global_dimensions, local_dimensions, 0, NULL, &event);
                clFinish(queue);
                clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &GPU_start, NULL);
                clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &GPU_stop, NULL);
                GPU_time += (GPU_stop - GPU_start);
                GPU_jumping_time += (GPU_stop - GPU_start);
                local_GPU_time += (GPU_stop - GPU_start);
                //auxiliary.iter_flag = !auxiliary.iter_flag;
                
                cout << "Iteration: " << i++ << "; Number of Edges: " << e << "; Number of Vertices: " << n << endl;
                //cout << "Time Taken (Hooking and Pointer Jumping - GPU): " << (double) local_GPU_time/1000000000 << "s" << endl;
            }
            //*/
        }
    }
    total_time = ((end - start) > ((double) GPU_time/1000000000)) ? (end - start) : (double) GPU_time/1000000000;
    
    cout << "Time Taken (DSU Algorithm CPU): " << (end - start) << "s" << endl;
    cout << "Time Taken (Hooking and Pointer Jumping - GPU): " << (double) GPU_time/1000000000 << "s" << endl;
    cout << "Time Taken (Pointer Jumping - GPU): " << (double) GPU_jumping_time/1000000000 << "s" << endl;
    
    GPU_time = 0;
    
    start = omp_get_wtime();
    process_cross_edges(visited, edge_list, cross_edges, number_of_edges);
    multilevel_pointer_jumping(visited, number_of_vertices, 0);
    end = omp_get_wtime();
    cout << "Time Taken (Cross Edges Processsing - CPU): " << (end - start) << "s" << endl;
    
    /*
     int i = 1;
     auxiliary.flag = false;
     clSetKernelArg(hooking_kernel, 3, sizeof(cross_edges_GPU), &cross_edges_GPU);
     e = number_of_edges;
     for(int i = 0; i < number_of_edges; i++)
     offset[i] = i;
     clSetKernelArg(hooking_kernel, 4, sizeof(cl_int), &e);
     clSetKernelArg(pointer_jumping_kernel, 2, sizeof(cl_int), &number_of_vertices);
     while(auxiliary.flag == false){
     //while(i == 1){
     auxiliary.flag = true;
     start_GPU = omp_get_wtime();
     e = accumulate(edge_list, offset, cross_edges, e);
     end_GPU = omp_get_wtime();
     GPU_time += (end_GPU - start_GPU) * 1000000000;
     clSetKernelArg(hooking_kernel, 4, sizeof(cl_int), &e);
     
     clEnqueueNDRangeKernel(queue, hooking_kernel, 1, NULL, global_dimensions, local_dimensions, 0, NULL, &event);
     clFinish(queue);
     clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &GPU_start, NULL);
     clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &GPU_stop, NULL);
     GPU_time += (GPU_stop - GPU_start);
     //multilevel_pointer_jumping(visited, number_of_vertices, 0);
     
     clEnqueueNDRangeKernel(queue, pointer_jumping_kernel, 1, NULL, global_dimensions, local_dimensions, 0, NULL, &event);
     clFinish(queue);
     clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &GPU_start, NULL);
     clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &GPU_stop, NULL);
     GPU_time += (GPU_stop - GPU_start);
     auxiliary.iter_flag = !auxiliary.iter_flag;
     
     cout << "Iteration: " << i++ << "; Number of Edges: " << e << endl;
     }
     //*/
    double temp_end = omp_get_wtime();
    total_time += (double) GPU_time/1000000000;
    
    cout << "Time Taken (Cross Edge Handling - GPU): " << (double) GPU_time/1000000000 << "s" << endl;
    //cout << "Total Time: " << total_time << "s" << endl;
    cout << "Total Time: " << temp_end - temp_start << "s" << endl;
    
    ///*
    cout << "Number of Elements is largest component: ";
    int temp = visited[0];
    int count = 0;
    for(int i = 0; i < number_of_vertices; i++)
        if(visited[i] == temp)
            count++;
    //cout << i << "->" << visited[i] << endl;
    //cout << "(" << edge_list[i].u << ", " << edge_list[i].v << ")" << endl;
    cout << count << endl;
    //*/
    
    /*
     for(int i = 0; i < number_of_edges; i++)
     cout << i << "->" << active_edges[i] << endl;
     */
    return 0;
}
