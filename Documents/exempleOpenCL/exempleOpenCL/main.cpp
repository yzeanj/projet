//
//  main.cpp
//  exempleOpenCL
//
//  Created by zijiang yang on 2018/4/5.
//  Copyright © 2018年 zijiang yang. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>

#ifdef __APPLE__
#include "OpenCL/cl.h"
#else
#include "CL/cl.h"
#endif

using namespace cv;
using namespace std;

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
    //获取平台信息
    for (int i = 0; i < numPlatforms; i++) {
        size_t size;
        
        errNum = clGetPlatformInfo(platform[i], CL_PLATFORM_VERSION, 0, NULL, &size);
        char *PInfo = (char*)malloc(size);
        errNum = clGetPlatformInfo(platform[i], CL_PLATFORM_VERSION, size, PInfo, NULL);
        
        printf("PlatFormInfo : %s\n",PInfo);
        free(PInfo);
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
    for (int i =0; i <num_device; i++) {
        size_t UnitNumber[3];
        clGetDeviceInfo(device[i], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t)*3, UnitNumber, NULL);
        for(int i = 0; i< 3; i++){
            printf("device max work group size in dimention %d is : %ld\n",i,UnitNumber[i]);
        }
        
    }
    
    // 在OpenCL平台上创建一个队列，先试GPU，再试CPU
    cl_context_properties contextProperties[] ={
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)(*platform),
        0
    };
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
    
    commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);
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

cl_bool ImageSupport(cl_device_id device) {
    cl_bool imageSupport = CL_FALSE;
    clGetDeviceInfo(device, CL_DEVICE_IMAGE_SUPPORT, sizeof(cl_bool),
                    &imageSupport, NULL);
    return imageSupport;
}

cl_mem LoadImage(cl_context context, std::string fileName, int &width, int &height){
    
    cv::Mat imageSrc = cv::imread(fileName);
    width = imageSrc.cols;
    height = imageSrc.rows;
    char *buffer = new char[width * height * 4];
    
    //数据传入方式：一个像素一个像素，按照B G R顺序，中间空一格 就像：
    // 12 237 34  221 88 99  22 33 99
    int w = 0;
    for (int v = height - 1; v >= 0; v--){
        for (int u = 0; u <width; u++){
            buffer[w++] = imageSrc.at<cv::Vec3b>(v, u)[0];
            buffer[w++] = imageSrc.at<cv::Vec3b>(v, u)[1];
            buffer[w++] = imageSrc.at<cv::Vec3b>(v, u)[2];
            w++;
        }
    }
    
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
                            buffer,&errNum);
    
    if (0 == clImage || CL_SUCCESS != errNum){
        std::cerr << "Error creating CL image object" << std::endl;
        return 0;
    }
    return clImage;
}

void Cleanup(cl_context context, cl_command_queue commandQueue,
             cl_program program, cl_kernel kernel,
             cl_mem imageObjects[2], cl_sampler sampler) {
    for (int i = 0; i < 2; i++) {
        if (imageObjects[i] != 0)
            clReleaseMemObject(imageObjects[i]);
    }
    if (commandQueue != 0)
        clReleaseCommandQueue(commandQueue);
    
    if (kernel != 0)
        clReleaseKernel(kernel);
    
    if (program != 0)
        clReleaseProgram(program);
    
    if (sampler != 0)
        clReleaseSampler(sampler);
    
    if (context != 0)
        clReleaseContext(context);
}

size_t RoundUp(int groupSize, int globalSize){
    int r = globalSize % groupSize;
    if (r == 0){
        return globalSize;
    } else {
        return globalSize + groupSize - r;
    }
}

void initTable(char* table16){
    table16[0] = '0';
    table16[1] = '1';
    table16[2] = '2';
    table16[3] = '3';
    table16[4] = '4';
    table16[5] = '5';
    table16[6] = '6';
    table16[7] = '7';
    table16[8] = '8';
    table16[9] = '9';
    table16[10] = 'a';
    table16[11] = 'b';
    table16[12] = 'c';
    table16[13] = 'd';
    table16[14] = 'e';
    table16[15] = 'f';
}

int main(int argc, char** argv){
    cl_context context = 0;
    
    cl_program program = 0;
    cl_device_id device = 0;
    cl_kernel kernel = 0;
    cl_mem imageObjects[3] = { 0, 0, 0 }; //一个原图像 一个目标图像
    cl_sampler sampler = 0;
    cl_int errNum;
    
    // get the device
    context = CreateContext();
    if (context == NULL) {
        cerr << "Failed to create OpenCL context." << endl;
        cin.get();
    }
    
    // 创建队列
    cl_command_queue commandQueue = 0;
    commandQueue = CreateCommandQueue(context, &device);
    if (commandQueue == NULL) {
        Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
        cin.get();
        return 1;
    }
    
    // is the device support this image.
    if (ImageSupport(device) != CL_TRUE){
        cerr << "OpenCL device does not support images." << endl;
        Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
        cin.get();
        return 1;
    }
    
    // 将图片载入OpenCL设备
    int width, height; //在LoadImage函数改变了其值
    string src0 = "./exempleOpenCL/myimage.jpg";
    imageObjects[0] = LoadImage(context, src0, width, height);
    if (imageObjects[0] == 0){
        cerr << "Error loading: " << string(src0) << endl;
        Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
        cin.get();
        return 1;
    }
    
    // 目标图像（缩小处理得到的）
    cl_image_format clImageFormat;
    clImageFormat.image_channel_order = CL_RGBA;
    clImageFormat.image_channel_data_type = CL_UNORM_INT8;
    cl_image_desc desc;
    //设置输出图片大小
    //set a number of group
    size_t resiTaill = 64;
    memset(&desc, 0, sizeof(desc));
    desc.image_height = resiTaill;
    desc.image_width = resiTaill;
    desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    imageObjects[1] = clCreateImage(context,CL_MEM_READ_WRITE,
                                    &clImageFormat, &desc,
                                    NULL, &errNum);
    if (errNum != CL_SUCCESS){
        cerr << "Error creating CL output image object." << endl;
        Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
        return 1;
    }
    
    // Create sampler for sampling image object
    sampler = clCreateSampler(context,
                              CL_FALSE, // Non-normalized coordinates
                              CL_ADDRESS_CLAMP_TO_EDGE,
                              CL_FILTER_NEAREST,
                              &errNum);
    
    if (errNum != CL_SUCCESS) {
        cerr << "Error creating CL sampler object." << endl;
        Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
        return 1;
    }
    
    // 创建函数项
    string cl_kernel_file = "./exempleOpenCL/exemple.cl";//OpenCL file path
    program = CreateProgram(context, device, cl_kernel_file);
    if (program == NULL) {
        Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
        cin.get();
        return 1;
    }
    
    //创建一个OpenCL中的函数
    kernel = clCreateKernel(program, "bgr2gray", NULL);
    if (kernel == NULL) {
        cerr << "Failed to create kernel" << endl;
        Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
        cin.get();
        return 1;
    }
    size_t workgroup_size;
    errNum = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE,
                                   sizeof(size_t), &workgroup_size, NULL);
    printf("KERNEL WORK GROUP SIZE is %ld\n",workgroup_size);
    
    
    float result[ resiTaill * resiTaill ];
    cl_mem memObjectResult = clCreateBuffer(context,
                                  CL_MEM_READ_WRITE,
                                  sizeof(float)*resiTaill * resiTaill, NULL, NULL);
    
    char* table_16 = (char*)malloc(sizeof(char)*16);
    initTable(table_16);
//    for(int i =0;i< 16;i++){
//        printf("%c",table_16[i]);
//    }
    cl_mem memTableRef = clCreateBuffer(context,
                                            CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                            sizeof(char)*16, table_16, NULL);
    
    char* ImageCode = (char*)malloc(sizeof(char)*resiTaill*resiTaill/4);
    cl_mem memImageCode = clCreateBuffer(context,
                                        CL_MEM_READ_WRITE,
                                        sizeof(char)*resiTaill*resiTaill/4, NULL, NULL);
    //set a size of one work group
    size_t localSize = 4;
    cl_mem memlocalSize = clCreateBuffer(context,
                                         CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                         sizeof(size_t), &localSize, NULL);
    size_t groupSize = resiTaill;
    cl_mem memgroupSize = clCreateBuffer(context,
                                         CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                         sizeof(size_t), &groupSize, NULL);
    
    
    // 传入参数
    errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &imageObjects[0]);
    errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &imageObjects[1]);
    errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &memObjectResult);
    errNum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &memTableRef);
    errNum |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &memImageCode);
    errNum |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &memlocalSize);
    errNum |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &memgroupSize);
    if (errNum != CL_SUCCESS) {
        cerr << "Error setting kernel arguments." << endl;
        Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
        return 1;
    }
    
    
    
    size_t localWorkSize[2] = {localSize,localSize};
    size_t globalWorkSize[2] = { RoundUp((int)localWorkSize[0], width),
                                    RoundUp((int)localWorkSize[1], height) };
    //commencer computation
    clock_t t1, t2;
    t1 = clock();
    errNum = clEnqueueNDRangeKernel(commandQueue, kernel,
                                    2, NULL,
                                    globalWorkSize, localWorkSize,
                                    0, NULL, NULL);
    
    if (errNum != CL_SUCCESS){
        cerr << "Error queuing kernel for execution." << endl;
        Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
        cin.get();
        return 1;
    }
    
    //计算机结果拷贝回主机
    errNum = clEnqueueReadBuffer(commandQueue, memImageCode, CL_TRUE, 0, sizeof(char)*(resiTaill*resiTaill/4),
                                 ImageCode, 0, NULL, NULL);
    if (errNum != CL_SUCCESS) {
        printf("Error reading result buffer.");
        Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
        return 1;
    }
    t2 = clock();
    //输出内核计算后的结构

    printf("%s",ImageCode);
    printf("\n");
    
    cout << "OpenCL - BGR2GRAY:----" << t2 - t1 << "ms" << endl;
    Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
    clReleaseMemObject(memObjectResult);
    clReleaseMemObject(memTableRef);
    clReleaseMemObject(memImageCode);
    clReleaseMemObject(memlocalSize);
    clReleaseMemObject(memgroupSize);
    free(table_16);
    free(ImageCode);
    return 0;
}
