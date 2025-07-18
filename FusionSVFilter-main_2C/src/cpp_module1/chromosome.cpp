/**
*消除kernel_cigar函数中的条件转移语句，向量化代码
*/
#include <pybind11/pybind11.h>
#include <iostream>
#include <algorithm>
#include <string>
#include <fstream>
#include <stdexcept>
#include <cmath>
#include <htslib/sam.h>
#include <htslib/hts.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>

#include <thread>
#include <mutex>
#include <condition_variable>

#include <vector>
#include <atomic>
#include <functional>
#include <chrono>
#include <hdf5.h>
#include <immintrin.h>//or #include <x86intrin.h>
using namespace std;
using namespace std::chrono;

namespace py = pybind11;

/*global variable*/
vector<int> ins_img;
vector<int> del_img;
vector<int> n_img;
string bam_path="",chromosome="";
int task_length[3]={0,0,0};
int non_empty=0;

long long global_extra_bam_op_time=0;
long long global_read_time=0;
	long long global_read_inner_time=0;
long long global_image_coding_time=0;
	long long global_kernel_cigar_time=0;

std::mutex mutex_iter;

////M:0,I:1,D:2,N:3,S:4,H:5,P:6,=:7,X:8
//int global_tau=1;
//int global_pixel[10]={63/global_tau, 191/global_tau, 127/global_tau, 0, 255/global_tau, 0, 0, 0, 0, 0};
//float global_begin_step[10]={0, -0.5, 0, 0, -0.5, 0, 0, 0, 0, 0};
//float global_end_step[10]={1, 0.5, 1, 0, 0.5, 0, 0, 0, 0, 0};
//int global_max_terminal_option[10]={1, 0, 1, 1, 0, 0, 0, 1, 1, 0};

// Represents a task in the queue
struct Task {
    std::function<void(void*)> function;
    void* arg;

    int begin,end;
	int zoom;
	int task_type;//用于区分不同的任务
	int task_id;

    Task* next;

    Task(){
		this->function=nullptr;
		this->arg=NULL;
		this->begin=0;
		this->end=0;
		this->zoom=0;
		this->task_type=0;
		this->task_id=0;
		this->next=NULL;
	}
};

struct ExtraArgs{
	samFile* task_sam_file;
	bam_hdr_t* task_header;
	hts_idx_t* task_idx;

	ExtraArgs(){
		this->task_sam_file=nullptr;
		this->task_header=nullptr;
		this->task_idx=nullptr;
	}

	ExtraArgs(samFile* sam_file,bam_hdr_t* header,hts_idx_t* idx){
		this->task_sam_file=sam_file;
		this->task_header=header;
		this->task_idx=idx;
	}
};

struct TaskArgs{
	Task* task;
	ExtraArgs* extra_param;

	TaskArgs(){
		this->task=nullptr;
		this->extra_param=nullptr;
	}

	TaskArgs(Task* task,ExtraArgs* extra_param){
		this->task=task;
		this->extra_param=extra_param;
	}
};

// Represents the thread pool
struct ThreadPool {
    Task* queueHead;
    Task* queueTail;

    std::vector<std::thread> threads;

    std::mutex mutex;
    std::mutex task_mutex;
    std::condition_variable cond;
    std::condition_variable cv;
    std::atomic<bool> quit{false};

	int thread_num_of_thread_pool;
    int task_num;
    int completed_tasks;

    ThreadPool(int thread_num,int task_num) {
        // Initialize the task queue
        queueHead = new Task();
        queueTail = queueHead;
        queueHead->next = nullptr;

		this->thread_num_of_thread_pool=thread_num;
		this->task_num=task_num;
		this->completed_tasks=0;

        // Create threads
        for (int i = 0; i < thread_num; ++i) {
            threads.emplace_back(&ThreadPool::threadRoutine, this);
        }
    }

    ~ThreadPool() {
        // Signal threads to quit
        quit.store(true);
        cond.notify_all();

        // Join all threads
        for (auto& thread : threads) {
            if (thread.joinable()) {
                thread.join();
            }
			//cout<<std::this_thread::get_id()<<"线程释放"<<endl;
        }

        // Clean up task queue
        while (queueHead) {
            Task* temp = queueHead;
            queueHead = queueHead->next;
            delete temp;
        }
    }

    void Resize_thread_num(int thread_num){
		for(int i=this->thread_num_of_thread_pool;i<thread_num;i++){
			threads.emplace_back(&ThreadPool::threadRoutine, this);
		}
		this->thread_num_of_thread_pool=thread_num;
    }

    void threadRoutine() {
		// 打开 BAM 文件
		samFile* thread_sam_file = sam_open(bam_path.c_str(), "rb");
		if (thread_sam_file == nullptr) {
			std::cerr << "Error opening BAM file." << std::endl;
			return;
		}

		bam_hdr_t* thread_header = sam_hdr_read(thread_sam_file);
		if (thread_header == nullptr) {
			std::cerr << "Error reading BAM header." << std::endl;
			sam_close(thread_sam_file);
			return;
		}

		hts_idx_t* thread_idx=sam_index_load(thread_sam_file,bam_path.c_str());
		if (thread_idx == nullptr) {
			std::cerr << "Error loading BAM index." << std::endl;
			bam_hdr_destroy(thread_header);
			sam_close(thread_sam_file);
			return;
		}
		ExtraArgs* thread_extra_args=new ExtraArgs(thread_sam_file,thread_header,thread_idx);

        while (true) {
            Task* task = nullptr;

            {
                std::unique_lock<std::mutex> lock(mutex);
                cond.wait(lock, [this] { return queueHead->next != nullptr || quit.load(); });

                if (quit.load()) {
                    break;
                }

                task = queueHead->next;
                if (task) {
                    queueHead->next = task->next;
                    if (queueHead->next == nullptr) {
                        queueTail = queueHead; // Update tail if queue is empty
                    }
                }
            }

            if (task) {
				//cout<<"线程:"<<std::this_thread::get_id()<<" 正在处理：chromosome:"<<chromosome<<" task_type:"<<task->task_type<<" task_id:"<<task->task_id<<"/"<<task_length[task->task_type-1]<<endl;
				TaskArgs* task_args=new TaskArgs(task,thread_extra_args);
                task->function(task_args);
                delete task_args->task;
                delete task_args;
            }

            {
				std::unique_lock<std::mutex> lock(task_mutex);
				completed_tasks++;
				cv.notify_all();
            }
        }
    }

    void addTask(std::function<void(void*)> func,int begin,int end,int task_type,int task_id) {
        Task* t = new Task();
        t->function = func;
        t->arg = t;
        t->begin=begin;
        t->end=end;
        t->zoom=1;
        t->task_type=task_type;
        t->task_id=task_id;
        t->next = nullptr;

        {
            // Lock the mutex for adding task
            std::lock_guard<std::mutex> lock(mutex);
            queueTail->next = t;
            queueTail = t;
        }

        cond.notify_one(); // Notify one waiting thread
    }
};

//双线性插值
std::vector<std::vector<int>> bilinear_interpolation(const std::vector<std::vector<int>>& M, int m1, int n1, int m2, int n2) {
	//cout<<std::this_thread::get_id()<<":bilinear_interpolation"<<endl;
    // 创建新的矩阵M1
    std::vector<std::vector<int>> M1(m2, std::vector<int>(n2));

    // 计算每个新像素对应的原像素的坐标
    float x_ratio = static_cast<float>(m1 - 1) / m2; // 原图高度与新图高度的比
    float y_ratio = static_cast<float>(n1 - 1) / n2; // 原图宽度与新图宽度的比

    for (int i = 0; i < m2; ++i) {
        for (int j = 0; j < n2; ++j) {
            // 原图中的位置
            float x = i * x_ratio;
            float y = j * y_ratio;

            int x1 = static_cast<int>(x); // 下取整
            int y1 = static_cast<int>(y); // 下取整
            int x2 = (x1 + 1 < m1) ? x1 + 1 : x1; // 上取整，确保不越界
            int y2 = (y1 + 1 < n1) ? y1 + 1 : y1; // 上取整，确保不越界

            // 计算插值
            float a = x - x1; // x的余量
            float b = y - y1; // y的余量

            // 双线性插值公式
            float value = (1 - a) * (1 - b) * M[x1][y1] +
                        a * (1 - b) * M[x2][y1] +
                        (1 - a) * b * M[x1][y2] +
                        a * b * M[x2][y2];
            M1[i][j]=int(std::round(value));
        }
    }

    return M1; // 返回新矩阵
}

/**
*普通版本的kernel_cigar
*和python代码运行结果对比，正确性没有问题
*/
//MS_0_I_127_D_255，加宽
/*
vector<int> kernel_cigar(bam1_t* read, int ref_min, int ref_max, int zoom){
	int w=int((ref_max-ref_min)/zoom);
	vector<int> row(w,0);
	int tau=1;
	int max_terminal=read->core.pos - ref_min;
	int begin=0,end=0;
	uint32_t* cigar=bam_get_cigar(read);
	for(int i=0;i<read->core.n_cigar;i++){
		uint32_t op=bam_cigar_op(cigar[i]);
		uint32_t len=bam_cigar_oplen(cigar[i]);
		//M:0,I:1,D:2,N:3,S:4,H:5,P:6,=:7,X:8
		if(op==0){//M
			begin=int(max_terminal/zoom);
			end=int((max_terminal+len)/zoom);
			max_terminal+=len;
			if(begin<0)begin=0;
			if(end>w)end=w;
			for(int j=begin;j<end;j++){
				row[j]=0/tau;
			}
		}
		else if(op==2){//D
			begin=int((max_terminal-len)/zoom);
			end=int((max_terminal+len)/zoom);
			max_terminal+=len;
			if(begin<0)begin=0;
			if(end>w)end=w;
			for(int j=begin;j<end;j++){
				row[j]=255/tau;
			}
		}
		else if(op==1){//I
			begin=int((max_terminal-len)/zoom);
			end=int((max_terminal+len)/zoom);
			max_terminal+=len;
			if(begin<0)begin=0;
			if(end>w)end=w;
			for(int j=begin;j<end;j++){
				row[j]=127/tau;
			}
		}
		else if(op==4){//S
			begin=int(max_terminal/zoom);
			end=int((max_terminal+len)/zoom);
			if(begin<0)begin=0;
			if(end>w)end=w;
			for(int j=begin;j<end;j++){
				row[j]=0/tau;
			}
		}
		else if(op==3||op==7||op==8){
			max_terminal+=len;
		}
	}
	vector<vector<int>> M;
	M.push_back(row);
	int m1=M.size();
	int n1=M[0].size();
	int m2=1;
	int n2=224;
	std::vector<std::vector<int>> M1=bilinear_interpolation(M,m1,n1,m2,n2);
	row=M1[0];
	return row;
}*/

//不加宽
vector<int> kernel_cigar(bam1_t* read, int ref_min, int ref_max, int zoom){
	int w=int((ref_max-ref_min)/zoom);
	vector<int> row(w,0);
	int tau=1;
	int max_terminal=read->core.pos - ref_min;
	int begin=0,end=0;
	uint32_t* cigar=bam_get_cigar(read);
	for(int i=0;i<read->core.n_cigar;i++){
		uint32_t op=bam_cigar_op(cigar[i]);
		uint32_t len=bam_cigar_oplen(cigar[i]);
		//M:0,I:1,D:2,N:3,S:4,H:5,P:6,=:7,X:8
		if(op==0){//M
			begin=int(max_terminal/zoom);
			end=int((max_terminal+len)/zoom);
			max_terminal+=len;
			if(begin<0)begin=0;
			if(end>w)end=w;
			for(int j=begin;j<end;j++){
				row[j]=0/tau;
			}
		}
		else if(op==2){//D
			begin=int((max_terminal-2*len)/zoom);
			end=int((max_terminal+2*len)/zoom);
			max_terminal+=len;
			if(begin<0)begin=0;
			if(end>w)end=w;
			for(int j=begin;j<end;j++){
				row[j]=255/tau;
			}
		}
		else if(op==1){//I
			begin=int((max_terminal-2*len)/zoom);
			end=int((max_terminal+2*len)/zoom);
			max_terminal+=len;
			if(begin<0)begin=0;
			if(end>w)end=w;
			for(int j=begin;j<end;j++){
				row[j]=127/tau;
			}
		}
		else if(op==4){//S
			begin=int((max_terminal)/zoom);
			end=int((max_terminal+len)/zoom);
			if(begin<0)begin=0;
			if(end>w)end=w;
			for(int j=begin;j<end;j++){
				row[j]=0/tau;
			}
		}
		else if(op==3||op==7||op==8){
			max_terminal+=len;
		}
	}
	vector<vector<int>> M;
	M.push_back(row);
	int m1=M.size();
	int n1=M[0].size();
	int m2=1;
	int n2=224;
	std::vector<std::vector<int>> M1=bilinear_interpolation(M,m1,n1,m2,n2);
	row=M1[0];
	return row;
}

int cigar_new_img_single_optimal(Task* task,ExtraArgs* task_extra_args){
	auto extra_bam_op_start=high_resolution_clock::now();
	int begin=task->begin;
	int end=task->end;
	int zoom=task->zoom;
	auto init_read_start=high_resolution_clock::now();
    bam1_t* read=bam_init1();
    if (read == nullptr) {
        std::cerr << "Error initializing BAM alignment." << std::endl;
        return 1;
    }
    auto init_read_end=high_resolution_clock::now();

	auto query_start=high_resolution_clock::now();
    //创建迭代器
    std::string region = chromosome + ":" + std::to_string(begin) + "-" + std::to_string(end);
    hts_itr_t* iter=sam_itr_querys(task_extra_args->task_idx,task_extra_args->task_header,region.c_str());
    if (iter == nullptr) {
        std::cerr << "Error creating iterator for region: " << region << std::endl;
        bam_destroy1(read);
        return 1;
    }
    auto query_end=high_resolution_clock::now();
    auto extra_bam_op_end=high_resolution_clock::now();

	auto read_start=high_resolution_clock::now();
    // 创建向量存储起始和结束位置
    vector<int> r_start;
    vector<int> r_end;
    vector<bam1_t*> read_vector;
	//遍历reads获取图像宽度
	int read_number=0;
	while(sam_itr_next(task_extra_args->task_sam_file,iter,read)>=0){
		read_number++;
		if(read_number>10000){
			cout<<"the read number of this begin-end pair great 10000."<<endl;
			break;
		}
		auto read_inner_start=high_resolution_clock::now();
		if (read->core.pos >= 0) {  // reference_start
			r_start.push_back(read->core.pos);
		}
		int end_pos = bam_endpos(read);
		if (end_pos >= 0) {  // reference_end
			r_end.push_back(end_pos);
		}
		read_vector.push_back(read);
		auto read_inner_end=high_resolution_clock::now();
		auto read_inner_time=duration_cast<microseconds>(read_inner_end-read_inner_start);
		global_read_inner_time+=static_cast<long long>(read_inner_time.count());

	}
	auto read_end=high_resolution_clock::now();

    vector<int>::iterator min_it = std::min_element(r_start.begin(), r_start.end());
	vector<int>::iterator max_it = std::max_element(r_end.begin(), r_end.end());

	int height=224;

	vector<vector<vector<int>>> cigar_img_3D;
	vector<vector<int>> cigar_img_2D;
	vector<int> row;

	//生成一张图片时间测量
	auto image_coding_start=high_resolution_clock::now();

	if(!r_start.empty()){
		non_empty++;
		int ref_min=*min_it;
		int ref_max=*max_it;

		int r_start_size=r_start.size();
		int read_lines=0;

		iter=sam_itr_querys(task_extra_args->task_idx,task_extra_args->task_header,region.c_str());
		while(sam_itr_next(task_extra_args->task_sam_file,iter,read)>=0){
			read_lines++;
			if(read_lines>10000){
				cout<<"the read number of this begin-end pair great 10000."<<endl;
				break;
			}
			auto kernel_cigar_start=high_resolution_clock::now();
			row = kernel_cigar(read, ref_min, ref_max, zoom);
			auto kernel_cigar_end=high_resolution_clock::now();
			auto kernel_cigar_time=duration_cast<microseconds>(kernel_cigar_end-kernel_cigar_start);
			global_kernel_cigar_time += static_cast<long long>(kernel_cigar_time.count());
			cigar_img_2D.push_back(row);
		}

		//for(int i=0;i<read_vector.size();i++){
		//	read_lines++;
		//	auto kernel_cigar_start=high_resolution_clock::now();
		//	row = kernel_cigar(read_vector[i], ref_min, ref_max, zoom);
		//	auto kernel_cigar_end=high_resolution_clock::now();
		//	auto kernel_cigar_time=duration_cast<microseconds>(kernel_cigar_end-kernel_cigar_start);
		//	global_kernel_cigar_time += static_cast<long long>(kernel_cigar_time.count());
		//	cigar_img_2D.push_back(row);
		//}
		if(read_lines<r_start_size){
			row=vector<int>(height,0);
			while(read_lines<r_start_size){
				cigar_img_2D.push_back(row);
				read_lines++;
			}
		}
		int m1=cigar_img_2D.size(),n1=cigar_img_2D[0].size();
		int m2=height,n2=height;
		cigar_img_2D=bilinear_interpolation(cigar_img_2D,m1,n1,m2,n2);
		cigar_img_3D.push_back(cigar_img_2D);

		if(task->task_type==1){//insert
            int r=cigar_img_3D[0].size();
            int c=cigar_img_3D[0][0].size();
            int height_square=height*height;
            for(int i=0;i<r;i++){
                for(int j=0;j<c;j++){
                    ins_img[task->task_id*height_square+i*height+j]=cigar_img_3D[0][i][j];
                }
            }
        }
        else if(task->task_type==2){//delete
            int r=cigar_img_3D[0].size();
            int c=cigar_img_3D[0][0].size();
            int height_square=height*height;
            for(int i=0;i<r;i++){
                for(int j=0;j<c;j++){
                    del_img[task->task_id*height_square+i*height+j]=cigar_img_3D[0][i][j];
                }
            }
        }
        else if(task->task_type==3){//negtive
            int r=cigar_img_3D[0].size();
            int c=cigar_img_3D[0][0].size();
            int height_square=height*height;
            for(int i=0;i<r;i++){
                for(int j=0;j<c;j++){
                    n_img[task->task_id*height_square+i*height+j]=cigar_img_3D[0][i][j];
                }
            }
        }
	}
	auto image_coding_end=high_resolution_clock::now();

	bam_destroy1(read);

	//计算时间
	auto extra_bam_op_time=duration_cast<microseconds>(extra_bam_op_end-extra_bam_op_start);
	auto read_time=duration_cast<microseconds>(read_end-read_start);
	auto image_coding_time=duration_cast<microseconds>(image_coding_end-image_coding_start);
    global_extra_bam_op_time+=static_cast<long long>(extra_bam_op_time.count());
    global_read_time+=static_cast<long long>(read_time.count());
    global_image_coding_time+=static_cast<long long>(image_coding_time.count());

	return 0;
}

// Task execution function
void taskFunc(void* arg) {
	TaskArgs* task_args=static_cast<TaskArgs*>(arg);
    Task* t = task_args->task;
    ExtraArgs* extra_args=task_args->extra_param;
    bool fail=true;
	int ans=0;
	while(fail){
		try{
			fail=false;
			ans=cigar_new_img_single_optimal(t,extra_args);
		}catch(const std::exception& e){
			fail=true;
			t->zoom+=1;
			std::cerr<<e.what()<<std::endl;
			std::cerr<<"Exception cigar_img_single_optimal() "<<chromosome<<" "<<t->zoom<<". The length = "<<(t->end-t->begin)<<std::endl;
		}
	}
}

void wait_for_completion(ThreadPool* pool) {
    std::unique_lock<std::mutex> lock(pool->task_mutex);
    pool->cv.wait(lock, [pool] { return pool->completed_tasks >= pool->task_num; });
}

void run_threads(string& ins_position_path,string& del_position_path,string& n_position_path,int ins_len,int del_len,int n_len,int& task_num,int& thread_num){
	//创建线程池
	// Create thread pool
    ThreadPool pool(thread_num,task_num);
	//printf("线程池创建完成\n");

	//读取ins_position,del_position,n_position
	//主线程往任务队列中添加任务，并且唤醒线程池中的线程
	//ins_position
	// 打开HDF5文件
    hid_t file_id = H5Fopen(ins_position_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);  // 修正函数名
    if (file_id < 0) {
        std::cerr << "无法打开文件:"<<ins_position_path<< std::endl;
        return;
    }

    // 打开存储数组的数据集
    hid_t dataset_id = H5Dopen(file_id, "array", H5P_DEFAULT);  // 修正函数名
    if (dataset_id < 0) {
        std::cerr << "无法打开数据集 array" << std::endl;
        H5Fclose(file_id);  // 修正函数名
        return;
    }

    // 获取数据集的维度信息
    hid_t dataspace_id = H5Dget_space(dataset_id);  // 修正函数名
    hsize_t dims[2];  // 假设是二维数组
    H5Sget_simple_extent_dims(dataspace_id, dims, nullptr);  // 修正函数名
    int ins_rows = dims[0];
    int ins_cols = dims[1];

    // 创建一个数组来存储读取的数据
    int ins_data[ins_rows][ins_cols];

    // 从数据集中读取数据
    H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, ins_data);  // 修正函数名

	// 关闭文件和数据集
    H5Dclose(dataset_id);  // 修正函数名
    H5Fclose(file_id);  // 修正函数名

    //delete
    // 打开HDF5文件
    file_id = H5Fopen(del_position_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);  // 修正函数名
    if (file_id < 0) {
        std::cerr << "无法打开文件:"<<del_position_path<< std::endl;
        return;
    }

    // 打开存储数组的数据集
    dataset_id = H5Dopen(file_id, "array", H5P_DEFAULT);  // 修正函数名
    if (dataset_id < 0) {
        std::cerr << "无法打开数据集 array" << std::endl;
        H5Fclose(file_id);  // 修正函数名
        return;
    }

    // 获取数据集的维度信息
    dataspace_id = H5Dget_space(dataset_id);  // 修正函数名
    H5Sget_simple_extent_dims(dataspace_id, dims, nullptr);  // 修正函数名
    int del_rows = dims[0];
    int del_cols = dims[1];

    // 创建一个数组来存储读取的数据
    int del_data[del_rows][del_cols];

    // 从数据集中读取数据
    H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, del_data);  // 修正函数名

	// 关闭文件和数据集
    H5Dclose(dataset_id);  // 修正函数名
    H5Fclose(file_id);  // 修正函数名

    //negative
    // 打开HDF5文件
    file_id = H5Fopen(n_position_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);  // 修正函数名
    if (file_id < 0) {
        std::cerr << "无法打开文件:"<<n_position_path<< std::endl;
        return;
    }

    // 打开存储数组的数据集
    dataset_id = H5Dopen(file_id, "array", H5P_DEFAULT);  // 修正函数名
    if (dataset_id < 0) {
        std::cerr << "无法打开数据集 array" << std::endl;
        H5Fclose(file_id);  // 修正函数名
        return;
    }

    // 获取数据集的维度信息
    dataspace_id = H5Dget_space(dataset_id);  // 修正函数名
    H5Sget_simple_extent_dims(dataspace_id, dims, nullptr);  // 修正函数名
    int n_rows = dims[0];
    int n_cols = dims[1];

    // 创建一个数组来存储读取的数据
    int n_data[n_rows][n_cols];

    // 从数据集中读取数据
    H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, n_data);  // 修正函数名

	// 关闭文件和数据集
    H5Dclose(dataset_id);  // 修正函数名
    H5Fclose(file_id);  // 修正函数名

	int begin,end;
	for(int i=0;i<ins_rows;i++){
		begin=ins_data[i][0];
		end=ins_data[i][1];
		pool.addTask(taskFunc,begin,end,1,i);
	}
	for(int i=0;i<del_rows;i++){
		begin=del_data[i][0];
		end=del_data[i][1];
		pool.addTask(taskFunc,begin,end,2,i);
	}
	for(int i=0;i<n_rows;i++){
		begin=n_data[i][0];
		end=n_data[i][1];
		pool.addTask(taskFunc,begin,end,3,i);
	}

	cout<<"ins_rows:"<<ins_rows<<" del_rows:"<<del_rows<<" n_rows:"<<n_rows<<endl;

    //测试点
    //cout<<"任务加载完成"<<endl;
    //计算时间测量
	auto compute_start=high_resolution_clock::now();
	wait_for_completion(&pool);
	auto compute_end=high_resolution_clock::now();
	auto compute_time=duration_cast<milliseconds>(compute_end - compute_start);

	//global_extra_bam_op_time=(1.0*global_extra_bam_op_time/task_num)/1000;
    //global_read_time=(1.0*global_read_time/task_num)/1000;
    //global_read_inner_time=(1.0*global_read_inner_time/task_num)/1000;
    //global_image_coding_time=(1.0*global_image_coding_time/task_num)/1000;
    //global_kernel_cigar_time=(1.0*global_kernel_cigar_time/task_num)/1000;

    global_extra_bam_op_time=(1.0*global_extra_bam_op_time)/1000;
    global_read_time=(1.0*global_read_time)/1000;
    global_read_inner_time=(1.0*global_read_inner_time);
    global_image_coding_time=(1.0*global_image_coding_time)/1000;
    global_kernel_cigar_time=(1.0*global_kernel_cigar_time)/1000;

	//输出时间
	cout<<"计算时间："<<compute_time.count()<<"ms"<<endl;
	cout<<"    额外操作bam文件的平均时间："<<global_extra_bam_op_time<<"ms"<<endl;
	cout<<"    计算read两个端点信息的平均时间："<<global_read_time<<"ms"<<endl;
	cout<<"    计算read两个端点信息循环内部的平均时间："<<global_read_inner_time<<"us"<<endl;
	cout<<"    图像编码平均时间："<<global_image_coding_time<<"ms"<<endl;
	cout<<"    kernel_cigar函数消耗的平均时间："<<global_kernel_cigar_time<<"ms"<<endl;

	//cout<<"即将销毁线程池"<<endl;

	return;
}

void __init(int ins_len,int del_len,int n_len){
	int height=224;
	task_length[0]=ins_len;task_length[1]=del_len;task_length[2]=n_len;
	ins_img=vector<int>(ins_len*1*height*height,0);
	del_img=vector<int>(del_len*1*height*height,0);
	n_img=vector<int>(n_len*1*height*height,0);
}

void save_to_h5(const string& ins_image_path,const string& del_image_path,const string& n_image_path,int ins_len,int del_len,int n_len) {
	int height=224;
	// insert
	// 创建并打开h5文件
    hid_t file_id = H5Fcreate(ins_image_path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
        std::cerr << "无法创建文件:"<<ins_image_path<< std::endl;
        return;
    }

    // 设置数据的维度
    hsize_t dims[4] = {ins_len, 1, height, height};
    hid_t dataspace_id = H5Screate_simple(4, dims, nullptr);

    // 创建数据集
    hid_t dataset_id = H5Dcreate(file_id, "array", H5T_NATIVE_INT, dataspace_id,
                                   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	// 将数据从 std::vector 转换为一个连续的内存块
	int* ins_data_ptr=ins_img.data();

    // 写入数据到文件
    H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, ins_data_ptr);

    // 关闭文件和数据集
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
    H5Fclose(file_id);

    //delete
    // 创建并打开h5文件
    file_id = H5Fcreate(del_image_path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
        std::cerr << "无法创建文件:"<<del_image_path<< std::endl;
        return;
    }

    // 设置数据的维度
    dims[0]=del_len;dims[1]=1;dims[2]=height;dims[3]=height;
    dataspace_id = H5Screate_simple(4, dims, nullptr);

    // 创建数据集
    dataset_id = H5Dcreate(file_id, "array", H5T_NATIVE_INT, dataspace_id,
                                   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	// 将数据从 std::vector 转换为一个连续的内存块
    // 这里我们直接获取指向数据的指针：&ins_img[0][0][0][0]
    //int* del_data_ptr = &del_img[0][0][0][0];
    int* del_data_ptr=del_img.data();

    // 写入数据到文件
    H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, del_data_ptr);

    // 关闭文件和数据集
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
    H5Fclose(file_id);

	//negative
    // 创建并打开h5文件
    file_id = H5Fcreate(n_image_path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
        std::cerr << "无法创建文件:"<<n_image_path<< std::endl;
        return;
    }

    // 设置数据的维度
    dims[0]=n_len;dims[1]=1;dims[2]=height;dims[3]=height;
    dataspace_id = H5Screate_simple(4, dims, nullptr);

    // 创建数据集
    dataset_id = H5Dcreate(file_id, "array", H5T_NATIVE_INT, dataspace_id,
                                   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	// 将数据从 std::vector 转换为一个连续的内存块
    // 这里我们直接获取指向数据的指针：&ins_img[0][0][0][0]
    //int* n_data_ptr = &n_img[0][0][0][0];
    int* n_data_ptr=n_img.data();

    // 写入数据到文件
    H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, n_data_ptr);

    // 关闭文件和数据集
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
    H5Fclose(file_id);
	/*
	if(chromosome=="1"){
		//正确性检查
		// insert
		nlohmann::json ins_json;// 创建 JSON 对象
		ins_json["ins_image"] = ins_img; // 将数组赋值给 JSON 对象
		string path="../data/json_image/"+chromosome+"/insert_cpp.json";
		// 打开文件并写入 JSON 数据
		std::ofstream ins_file(path);
		if (ins_file.is_open()) {
			ins_file << ins_json.dump(4); // 格式化输出为 4 个空格缩进
			ins_file.close();
			//cout << "Data saved to " << path << endl;
		} else {
			cerr << "Could not open file: " << path << endl;
		}
	}*/
}

void solve(string& ins_position_path,string& del_position_path,string& n_position_path,string& ins_image_path,string& del_image_path,string& n_image_path,int ins_len,int del_len,int n_len,string& bam_path1,string& chromosome1,int task_num,int thread_num){
	non_empty=0;

	global_extra_bam_op_time=0;
	global_read_time=0;
	global_read_inner_time=0;
	global_image_coding_time=0;
	global_kernel_cigar_time=0;

	bam_path=bam_path1;
	chromosome=chromosome1;
	cout<<"## cpp model:"<<"bam_path: "<<bam_path<<" chromosome: "<<chromosome<<endl;
	__init(ins_len,del_len,n_len);
	//__init_bam_file(bam_path1);
	run_threads(ins_position_path,del_position_path,n_position_path,ins_len,del_len,n_len,task_num,thread_num);
	//__destroy_bam_file();
	auto store_data_start=high_resolution_clock::now();
	save_to_h5(ins_image_path,del_image_path,n_image_path,ins_len,del_len,n_len);
	auto store_data_end=high_resolution_clock::now();
	auto store_data_time=duration_cast<milliseconds>(store_data_end - store_data_start);
	cout<<"存储数据时间："<<store_data_time.count()<<"ms"<<endl;
	cout<<"chromosome:"<<chromosome<<" non_empty:"<<non_empty<<endl;
}

PYBIND11_MODULE(chr_module,m){
	m.def("run_threads",&solve,"A function that solve problem.");
}