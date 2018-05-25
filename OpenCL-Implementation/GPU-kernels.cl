#define max(a, b) (a > b) ? a : b
#define min(a, b) (a < b) ? a : b

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

kernel void initialize(global int* visited, int n, global struct Auxiliary* auxiliary){
	int id = get_global_id(0);
	for(int i = id; i < n; i+= auxiliary->num_threads)
		visited[i] = i;
}

kernel void hook(global int* visited, global struct Edge* edge_list, global int* offset, global bool* active_edges, int e, global struct Auxiliary* auxiliary){
	int id = get_global_id(0);
	int u, v, index;
	int Mx, Mn;//, mx, mn;
	for(int i = id; i < e; i += auxiliary->num_threads){
        index = offset[i];
		u = edge_list[index].u;
		v = edge_list[index].v;
		if(active_edges[index] && (visited[u] != visited[v])){
			auxiliary->flag = false;
			Mx = max(visited[u], visited[v]);
			Mn = min(visited[u], visited[v]);
			//if(auxiliary->iter_flag == false)
                visited[Mn] = Mx;
			//else
			//	visited[Mx] = Mn;
		}
		else
			active_edges[index] = false;
	}
}

kernel void multi_pointer_jumping(global int* visited, int n, global struct Auxiliary* auxiliary, int offset){
    int id = get_global_id(0);
    int lid = get_local_id(0);
    int y, x;
    __local bool flag;
    if(lid == 0)
        flag = false;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = id; i < n; i += auxiliary->num_threads){
        y = visited[i + offset];
        x = visited[y];
        if(x != y){
            flag = true;
            visited[i + offset] = x;
        }
    }
    if(lid == 0)
        if(flag)
            auxiliary->pointer_jumping_flag = true;
}

kernel void update_mask(global int* visited, global int* node_type, int n, global struct Auxiliary* auxiliary, int offset){
    int id = get_global_id(0);
    for(int i = id; i < n; i += auxiliary->num_threads){
        if(visited[i + offset] == (i + offset))
            node_type[i + offset] = 0;
        else
            node_type[i + offset] = 1;
    }
}

kernel void multi_pointer_jumping_masked(global int* visited, global int* node_type, global int* offset, int n, global struct Auxiliary* auxiliary){
    int id = get_global_id(0);
    int lid = get_local_id(0);
    int y, x, index;
    __local bool flag;
    if(lid == 0)
        flag = false;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = id; i < n; i += auxiliary->num_threads){
        index = offset[i];
        if(node_type[index] == 0){
            y = visited[index];
            x = visited[y];
            if(x != y){
                flag = true;
                visited[index] = x;
            }
            else
                node_type[index] = -1;
        }
    }
    if(lid == 0)
        if(flag)
            auxiliary->pointer_jumping_flag = true;
}

kernel void multi_pointer_jumping_unmasked(global int* visited, global int* node_type, global int* offset, int n, global struct Auxiliary* auxiliary){
    int id = get_global_id(0);
    int lid = get_local_id(0);
    int y, x;
    for(int i = id; i < n; i += auxiliary->num_threads){
        if(node_type[i] == 1){
            y = visited[i];
            x = visited[y];
            if(x != y)
                visited[i] = x;
        }
    }
}

/*
kernel void multi_pointer_jumping(global int* visited, global int* temp, int n, global struct Auxiliary* auxiliary, int offset){
	int id = get_global_id(0);
	bool flag;
	while(true){
		flag = false;
		for(int i = id; i < n; i += auxiliary->num_threads){
			temp[i] = visited[visited[i + offset]];
			if(temp[i] != visited[i + offset])
				flag = true;
		}
		if(!flag)
			break;
		else
			for(int i = id; i < n; i += auxiliary->num_threads)
				visited[i + offset] = temp[i];
	}
			
}
*/
 
kernel void process_cross_edges(global int* visited, global struct Edge* edge_list, global bool* active_edges, int e, global struct Auxiliary* auxiliary){
    int id = get_global_id(0);
    int u, v;
    int Mx, Mn;//, mx, mn;
    for(int i = id; i < e; i += auxiliary->num_threads){
        u = edge_list[i].u;
        v = edge_list[i].v;
        if(active_edges[i] && (visited[u] != visited[v])){
            auxiliary->flag = false;
            //Mx = max(visited[u], visited[v]);
            Mn = min(visited[u], visited[v]);
            visited[visited[u]] = Mn;
            visited[visited[v]] = Mn;
        }
        else
            active_edges[i] = false;
    }
}

