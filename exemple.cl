const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

__kernel void bgr2gray(read_only image2d_t sourceImage,
                       write_only image2d_t targetImage,
                       __global float *globalBufSum,
                       __global const char* table16Ref,
                       __global char* CodeImage) {
    
    int2 pos = {(int)get_global_id(0), (int)get_global_id(1)};
    
    const float sourceW = get_image_width(sourceImage);
    const float sourceH = get_image_height(sourceImage);
    
    const float targW = get_image_width(targetImage);
    const float targH = get_image_height(targetImage);
    
    if (pos.x >= targW || pos.y >= targH)
        return;
    float scaleX = 1.0f*sourceW/targW;
    float scaleY = 1.0f*sourceH/targH;
        
    float2 spos = (float2) (pos.x*scaleX, pos.y*scaleY);
    
    float4 v = 0;
    //int iscaleX=ceil(scaleX);
    //int iscaleY=ceil(scaleY);
    
    for (int y=0;y<scaleY;y++) {
        for (int x=0;x<scaleX;x++) {
            v += read_imagef(sourceImage, sampler, spos + (int2) {spos.x,spos.y});
        }
    }
    v = v/(scaleX*scaleY);
    float gray=(0.2989 * v.x + 0.5870 * v.y + 0.1140 * v.z);
    v= (float4) {gray,gray,gray,0};
    write_imagef(targetImage, pos, v);
 
    int indexArray = int(pos.x + targW * pos.y);
    globalBufSum[indexArray] = gray;
    
    float Sumbufrun;
    if(pos.x ==0 && pos.y ==0){
        Sumbufrun = 0.0f;
        for(int i =0; i < (targW * targH); i++){
            Sumbufrun = Sumbufrun + globalBufSum[i];
        }
        //printf("bufele == %f!\n",Sumbufrun);
        Sumbufrun = Sumbufrun /(targW * targH);
        
        int sumbufRun = 0;
        int sign = 0;
        for(int j =0; j < (targW * targH); j++){
            if(sign == 4){
                //printf("====code number %d---- is %c\n",j,table16Ref[sumbufRun]);
                CodeImage[(j/4)-1] = table16Ref[sumbufRun];
                //printf("====code number %d---- is %c\n",(i/4)-1,CodeImage[(i/4)-1]);
                sign = 0;
                sumbufRun = 0;
            }
            if(globalBufSum[j]>Sumbufrun){
                if(sign != 0){
                    sumbufRun = sumbufRun<<1;
                    }
                sumbufRun = sumbufRun | 1;
            }else{
                if(sign != 0){
                    sumbufRun = sumbufRun<<1;
                }
            }
            sign++;
        }
    }
}
