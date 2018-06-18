//
//  main.cpp
//  exempleOpenCL
//
//  Created by zijiang yang on 2018/4/5.
//  Copyright © 2018年 zijiang yang. All rights reserved.
//

#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>
#include <string>
#include <vector>

#include<opencv2/opencv.hpp>
#include<opencv2/core/ocl.hpp>
#include<FreeImage.h>
#include<FreeImagePlus.h>

#ifdef __APPLE__
#include "OpenCL/cl.h"
#else
#include "CL/cl.h"
#endif

using namespace cv;
using namespace std;
clock_t t_transmettre;
clock_t t_calcule;
clock_t t_lecture;

cl_context CreateContext(){
    cl_int errNum;
    cl_uint numPlatforms;
    cl_platform_id *platform;
    cl_context context = NULL;
    
    // 选择OpenCL平台
    errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
    
    platform = (cl_platform_id*)malloc(sizeof(cl_platform_id)*numPlatforms);
    
    errNum = clGetPlatformIDs(numPlatforms, platform, NULL);
    if (errNum!= CL_SUCCESS || numPlatforms <= 0) {
        printf("fail to find any OpenCL platform");
        return NULL;
    }
    
    //获得平台设备,使用GPU作为平台设备
    cl_uint num_device;
    cl_device_id *device;
    errNum = clGetDeviceIDs(platform[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_device);
    device = (cl_device_id *)malloc(sizeof(cl_device_id)*(num_device));
    //初始化可用的设备
    errNum = clGetDeviceIDs(platform[0], CL_DEVICE_TYPE_GPU, num_device, device, NULL);
    if (errNum != CL_SUCCESS) {
        printf("there is no GPU,trying CPU...");
        errNum = clGetDeviceIDs(*platform, CL_DEVICE_TYPE_CPU, num_device, device, NULL);
    }
    if (errNum != CL_SUCCESS) {
        printf("there is no GPU and CPU...");
        return NULL;
    }
    
    // 在OpenCL平台上创建一个队列，先试GPU，再试CPU
    cl_context_properties contextProperties[] = {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)(*platform),0};
    
    context = clCreateContextFromType(contextProperties,
                                      CL_DEVICE_TYPE_GPU,
                                      NULL, NULL, &errNum);
    if (errNum != CL_SUCCESS){
        std::cout << "Could not create GPU context, trying CPU..." << std::endl;
        context = clCreateContextFromType(contextProperties,
                                          CL_DEVICE_TYPE_CPU,
                                          NULL, NULL, &errNum);
        if (errNum != CL_SUCCESS){
            std::cerr << "Failed to create an OpenCL GPU or CPU context." << std::endl;
            return NULL;
        }
    }
    return context;
}

cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device){
    cl_int errNum;
    cl_device_id *devices;
    cl_command_queue commandQueue = NULL;
    size_t deviceBufferSize = -1;
    
    // First get the size of the devices buffer
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
    if (errNum != CL_SUCCESS){
        std::cerr << "Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)";
        return NULL;
    }
    
    if (deviceBufferSize <= 0){
        std::cerr << "No devices available.";
        return NULL;
    }
    
    // Allocate memory for the devices buffer
    devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
    if (errNum != CL_SUCCESS){
        std::cerr << "Failed to get device IDs";
        return NULL;
    }
    
    // In this example, we just choose the first available device. In a
    // real program, you would likely use all available devices or choose
    // the highest performance device based on OpenCL device queries
    commandQueue = clCreateCommandQueue(context, devices[0], 0, &errNum);
    if (commandQueue == NULL){
        std::cerr << "Failed to create commandQueue for device 0";
        return NULL;
    }
    
    *device = devices[0];
    delete[] devices;
    return commandQueue;
}

//  Create an OpenCL program from the kernel source file
cl_program CreateProgram(cl_context context, cl_device_id device, std::string fileName){
    cl_int errNum;
    cl_program program;
    
    std::ifstream kernelFile(fileName, std::ios::in);
    if (!kernelFile.is_open()){
        std::cerr << "Failed to open file for reading: " << fileName << std::endl;
        return NULL;
    }
    
    std::ostringstream oss;
    oss << kernelFile.rdbuf();
    
    std::string srcStdStr = oss.str();
    const char *srcStr = srcStdStr.c_str();
    program = clCreateProgramWithSource(context, 1,
                                        (const char**)&srcStr,
                                        NULL, NULL);
    if (program == NULL) {
        std::cerr << "Failed to create CL program from source." << std::endl;
        return NULL;
    }
    
    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS) {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              sizeof(buildLog), buildLog, NULL);
        
        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
        clReleaseProgram(program);
        return NULL;
    }
    return program;
}

cl_mem LoadImage(cl_context context, std::string fileName, int &width, int &height){
    
    
    cv::Mat imageSrc = cv::imread(fileName);
    width = imageSrc.cols;
    height = imageSrc.rows;
   
    // Create OpenCL image
    cl_image_format clImageFormat;
    clImageFormat.image_channel_order = CL_RGBA;
    clImageFormat.image_channel_data_type = CL_UNORM_INT8;
    
    cl_image_desc clImageDesc;
    memset(&clImageDesc, 0, sizeof(cl_image_desc));
    clImageDesc.image_type =CL_MEM_OBJECT_IMAGE2D;
    clImageDesc.image_width = width;
    clImageDesc.image_height = height;
    
    cl_int errNum;
    cl_mem clImage = clCreateImage(context,
                            CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                            &clImageFormat,&clImageDesc,
                            imageSrc.ptr(),&errNum);
    
    if (0 == clImage || CL_SUCCESS != errNum){
        std::cerr << "Error creating CL image object" << std::endl;
        return 0;
    }
    return clImage;
}

void Cleanup(cl_context context, cl_device_id device, cl_program program,
             cl_command_queue commandQueue, cl_kernel kernel) {
    if (commandQueue != 0)
        clReleaseCommandQueue(commandQueue);
    if (kernel != 0)
        clReleaseKernel(kernel);
    if (program != 0)
        clReleaseProgram(program);
    if (device != 0)
        clReleaseDevice(device);
    if (context != 0)
        clReleaseContext(context);
}

cl_context platFormeInial(cl_command_queue& commandQueue, cl_kernel& kernel){
    cl_context context = 0;
    cl_device_id device = 0;
    cl_program program = 0;
    
    // get the device
    context = CreateContext();
    if (context == NULL) {
        Cleanup(context,device,program,commandQueue,kernel);
        cerr << "Failed to create OpenCL context." << endl;
        cin.get();
        return NULL;
    }
    
    // creat command queue
    commandQueue = CreateCommandQueue(context, &device);
    if (commandQueue == NULL) {
        Cleanup(context,device,program,commandQueue,kernel);
        cerr << "Failed to create OpenCL commande et device." << endl;
        cin.get();
        return NULL;
    }
    
    // creat kernel program
    string cl_kernel_file = "./exemple.cl";//OpenCL file path
    program = CreateProgram(context, device, cl_kernel_file);
    if (program == NULL) {
        Cleanup(context,device,program,commandQueue,kernel);
        cerr << "Failed to create OpenCL commande et device." << endl;
        cin.get();
        return NULL;
    }
    
    //creat kernel foction
    kernel = clCreateKernel(program, "bgr2gray", NULL);
    if (kernel == NULL) {
        Cleanup(context,device,program,commandQueue,kernel);
        cerr << "Failed to create OpenCL commande et device." << endl;
        cin.get();
        return NULL;
    }
    return context;
}

cl_int InitialImageObjet(cl_context context,cl_mem *imageObjects,size_t resiTaill,
                         cl_mem& memObjectResult,cl_mem& memImageCode){
    cl_int errNum;
    //cl_mem imageObjects[2] = {0, 0}; //一个原图像 一个目标图像
    
    // 目标图像（缩小处理得到的）
    cl_image_format clImageFormat;
    clImageFormat.image_channel_order = CL_RGBA;
    clImageFormat.image_channel_data_type = CL_UNORM_INT8;
    cl_image_desc desc;
    
    //设置输出图片大小
    memset(&desc, 0, sizeof(desc));
    desc.image_height = resiTaill;
    desc.image_width = resiTaill;
    desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    imageObjects[1] = clCreateImage(context,CL_MEM_READ_WRITE,
                                    &clImageFormat, &desc,
                                    NULL, &errNum);
    //float* result = (float*)malloc(sizeof(float)*resiTaill * resiTaill);
    memObjectResult = clCreateBuffer(context,
                                            CL_MEM_READ_WRITE,
                                            sizeof(float)*resiTaill * resiTaill, NULL, NULL);
    
    memImageCode = clCreateBuffer(context,
                                         CL_MEM_READ_WRITE,
                                         sizeof(char)*resiTaill*resiTaill, NULL, NULL);
    return errNum;
}

void ReadFileNameInFile(string sfilePicture,vector<string>& fileList){
    DIR* dir;
    struct dirent* ptr;
    dir = opendir(sfilePicture.c_str());
    while((ptr = readdir(dir)) != NULL){
        if(strcmp(ptr->d_name,".") != 0 && strcmp(ptr->d_name,"..") != 0){
            //cout<<"filename = "<<ptr->d_name<<endl;
            if(strcmp(ptr->d_name,".DS_Store") != 0){
                fileList.push_back(ptr->d_name);
            }
        }
    }
    
    closedir(dir);
}

cl_int traitImage(cl_context context,cl_command_queue &commandQueue,cl_kernel &kernel,string sfilePicture){
    
    size_t resiTaill = 10; //size de image genere heith = withe
    cl_mem imageObjects[2] = {0, 0};
    cl_mem memObjectResult;
    cl_mem memImageCode;
    //initiale les buffer de image
    cl_int errNum = InitialImageObjet(context,imageObjects,resiTaill,memObjectResult,memImageCode);
    if (errNum != CL_SUCCESS){
        Cleanup(context,NULL,NULL,commandQueue,kernel);
        cerr << "Error creating CL output image object." << endl;
        return errNum;
    }
    
    vector<string> fileList;
    ReadFileNameInFile(sfilePicture,fileList);
    
    //boucle pour traiter image
    for(int i_image = 0; i_image < fileList.size(); i_image++){
        int width, height;
        clock_t t1,t2,t3,t4;
        string fileJpgName = fileList.at(i_image);
        string src0 = sfilePicture+ fileJpgName;
        clock_t t_lecture1 = clock();//Horodatages lecture begin
        imageObjects[0] = LoadImage(context, src0, width, height);
        clock_t t_lecture2 = clock();;//Horodatages lecture end
        t_lecture = t_lecture + (t_lecture2 - t_lecture1);
        if (imageObjects[0] == 0){
            cerr << "Error loading: " << string(src0) << endl;
            Cleanup(context,NULL,NULL,commandQueue,kernel);
            cin.get();
            return errNum;
        }
        t1 = clock();//Horodatages1
        // 传入参数
        errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &imageObjects[0]);
        errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &imageObjects[1]);
        errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &memObjectResult);
        errNum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &memImageCode);
        if (errNum != CL_SUCCESS) {
            cerr << "Error setting kernel arguments." << endl;
            Cleanup(context,NULL,NULL,commandQueue,kernel);
            cin.get();
            return errNum;
        }
        t2 = clock();//Horodatages2
        size_t globalWorkSize[2] = {resiTaill,resiTaill};
        errNum = clEnqueueNDRangeKernel(commandQueue, kernel,
                                        2, NULL,
                                        globalWorkSize, NULL,
                                        0, NULL, NULL);
        
        if (errNum != CL_SUCCESS){
            cerr << "Error queuing kernel for execution." << endl;
            Cleanup(context,NULL,NULL,commandQueue,kernel);
            cin.get();
            return errNum;
        }
        t3 = clock();//Horodatages3
        //计算机结果拷贝回主机
        char* ImageCode = (char*)malloc(sizeof(char)*resiTaill*resiTaill);
        errNum = clEnqueueReadBuffer(commandQueue, memImageCode, CL_TRUE, 0, sizeof(char)*(resiTaill*resiTaill),
                                     ImageCode, 0, NULL, NULL);
        if (errNum != CL_SUCCESS) {
            printf("Error reading result buffer.");
            Cleanup(context,NULL,NULL,commandQueue,kernel);
            cin.get();
            return errNum;
        }
        t4 = clock();//Horodatages4
        t_transmettre = t_transmettre + (t2 - t1)+(t4 - t3);
        t_calcule = t_calcule+ (t3 - t2);
        //resault of image code
//        for (long i = resiTaill * resiTaill -1; i >= 0; i--) {
//            printf("%c",ImageCode[i]);
//        }
//        printf("\n");
        
    }
    return errNum;
}

int main(int argc, char** argv){
    
    cl_context my_context = 0;
    cl_command_queue my_commandQueue = 0;
    cl_kernel my_kernel = 0;
    //create plat-forme
    my_context = platFormeInial(my_commandQueue,my_kernel);
    
    //string srcPicFile = "./picture1/";
    string srcPicFile = argv[1];
    
    //traitement d'image
    cl_int errNum = traitImage(my_context,my_commandQueue,my_kernel,srcPicFile);
    
    if (errNum != CL_SUCCESS) {
        printf("program is Error");
        Cleanup(my_context,NULL,NULL,my_commandQueue,my_kernel);
    }
    //show the diffrents time
    if (argv[2] != NULL) {
        string srcTimeType(argv[2]);
        string strtrans ="trans";
        string srccacul ="calcul";
        string srclecture ="lecture";
        if (srcTimeType == strtrans) {
            cout<<"transition time: "<<static_cast<double>(t_transmettre)/CLOCKS_PER_SEC*1000<<"ms"<<endl;
        }
        if (srcTimeType == srccacul) {
            cout<<"calcul time: "<<static_cast<double>(t_calcule)/CLOCKS_PER_SEC*1000<<"ms"<<endl;
        }
        if (srcTimeType == srclecture) {
            cout<<"lecture time: "<<static_cast<double>(t_lecture)/CLOCKS_PER_SEC*1000<<"ms"<<endl;
        }
    }
    
    return 0;
}
