const sampler_t sampler = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void bgr2gray(read_only image2d_t sourceImage,
                       write_only image2d_t targetImage,
                       __global float *globalBufSum,
                       __global const char* table16Ref,
                       __global char* CodeImage,
                       __global const int* globalLocalsize,
                       __global const int* globalGroupsize) {
    
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    
    const float sw = get_image_width(sourceImage);
    const float sh = get_image_height(sourceImage);
    
    if (x >= sw || y >= sh)
        return;
    
    //const float w = get_image_width(targetImage);
    //const float h = get_image_height(targetImage);
    //float2 coord = { ((float) x / w) * sw,((float) y / h) * sh };
    
    float2 coord = {(float)x,(float)y};
    float4 pixel = read_imagef(sourceImage,sampler,coord);
    
    float dst_ = 0.11 * pixel.x + 0.59 * pixel.y + 0.30 * pixel.z;
    
    int xNumGroup = get_group_id(0);
    int yNumGroup = get_group_id(1);
    
    int xNumlocal = get_local_id(0);
    int yNumlocal = get_local_id(1);
    
    //将灰度值加入新数组中
    __local float bufSum[4*4];
    bufSum[xNumlocal+(*globalLocalsize)*yNumlocal] = dst_;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    //数组求和
    __local float Sumbufrun;
    if(xNumlocal ==0 && yNumlocal ==0){
        Sumbufrun = 0.0f;
        for(int i =0; i < (*globalLocalsize)*(*globalLocalsize); i++){
            Sumbufrun = Sumbufrun + (bufSum[i]/((*globalLocalsize)*(*globalLocalsize)));
        }
        globalBufSum[xNumGroup + (*globalGroupsize)*yNumGroup] = Sumbufrun;
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
    //write_imagef(targetImage, (int2)(xNumGroup, yNumGroup), (float4)(dst_,dst_,dst_,1.0f));
    //输出新的灰度值图片
    write_imagef(targetImage, (int2)(xNumGroup, yNumGroup), (float4)(globalBufSum[xNumGroup + (*globalGroupsize)*yNumGroup],
                                                                     globalBufSum[xNumGroup + (*globalGroupsize)*yNumGroup],
                                                                     globalBufSum[xNumGroup + (*globalGroupsize)*yNumGroup],
                                                                     1.0f));
    
    if(x ==0 && y ==0){
        float Sumbufloc = 0.0f;
        for(int j =0; j < (*globalGroupsize)*(*globalGroupsize); j++){
            Sumbufloc = Sumbufloc + globalBufSum[j]/((*globalGroupsize)*(*globalGroupsize));
            barrier(CLK_GLOBAL_MEM_FENCE);
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
        //printf("test sumbu %f\n",Sumbufloc);

        int sumbufRun = 0;
        int sign = 0;
        for(int i = 0; i<= (*globalGroupsize)*(*globalGroupsize); i++){

            if(sign == 4){
                //printf("====code number %d---- is %c\n",i,table16Ref[sumbufRun]);
                CodeImage[(i/4)-1] = table16Ref[sumbufRun];
                printf("====code number %d---- is %c\n",(i/4)-1,CodeImage[(i/4)-1]);
                sign = 0;
                sumbufRun = 0;
            }
            if(globalBufSum[i]>Sumbufloc){
                if(sign!=0){
                    sumbufRun = sumbufRun<<1;
                }
                sumbufRun = sumbufRun | 1;

            }else{
                if(sign!=0){
                    sumbufRun = sumbufRun<<1;
                }
            }
            sign++;
        }
    }
    
    //write_imagef(targetImage, (int2)(xNumGroup,yNumGroup), (float4)Sumbufrun);
}
