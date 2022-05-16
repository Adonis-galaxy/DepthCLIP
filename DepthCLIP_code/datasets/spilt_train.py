import os
data=open("/home/rrzhang/zengzy/code/clip_depth/datasets/nyudepthv2_test_files_with_gt_dense.txt",'w+') 


file_big = '/home/rrzhang/zengzy/code/clip_depth/datasets/NYU_Depth_V2/official_splits/test'
file_name = os.listdir(file_big)
file_name=sorted(file_name)
for j in file_name:
    name = os.listdir(file_big+'/'+j)
    name=sorted(name)
    name_1=name[1:len(name)//2+1]
    name_2=name[len(name)//2+1:]
    for i in range(len(name_1)):
        print(j+'/'+name_1[i],j+'/'+name_2[i],file=data)
        
data.close()