# -*- coding: utf-8 -*-
"""
region_growing.py
Created on Fri Dec  1 17:10:53 2017

@author: Cho
Ref 
[1] https://kr.mathworks.com/matlabcentral/fileexchange/19084-region-growing?focused=5098324&tab=function
[2] Broadband ultrasound attenuation imaging: algorithm development and clinical assessment of a region growing technique
"""
import numpy as np
import cv2 

def regiongrowing(I, x, y, reg_maxdist):
    # 초기 변수 
    J = np.zeros_like(I) # region이 저장될 배열 
    
    reg_mean = I[y, x] # 영역 평균 명암 
    reg_size = 1 # 영역 화소 수 
    
    # 확장된 영역의 주변 화소를 저장할 메모리 공간 할당  
    neg_free = 10000
    neg_pos = 0
    neg_list = np.zeros((neg_free, 3))    

    pixdist = 0 # 새로운 추가된 영역의 픽셀과 영역 평균 명암과의 거리 

    neighb = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    #%% 영역 후보 화소와 영역(region) 사이의 거리가 임계값 보다 높을 때까지 반복  
    while(pixdist < reg_maxdist and reg_size < I.size):
    
        # 새로운 주변 화소 추가 
        for j in range(4):
            xn = x + neighb[j][0]
            yn = y + neighb[j][1]
            
            # 주변 화소가 이미지 영역 안에 범위인지 확인 
            ins = xn >= 0 and yn >=0 and xn <= I.shape[1] and yn < I.shape[0]
            
            # 주변화소가 영역 안이고 아직 영역으로 확장되지 않았으면, 주변 화소로 추가 
            if(ins and J[yn, xn] == 0):
                neg_pos = neg_pos + 1 # 주변화소 리스트 접근 인덱스 추가 
                neg_list[neg_pos, :] = [xn, yn, I[yn, xn]] # 추가된 화소의 인덱스와 명암을 리스트에 추가 
                J[yn, xn] = 1 # 현재 화소를 검토된 영역으로 만듬 
        
        # 메모리 공간 확장         
        if(neg_pos + 10 > neg_free): 
            np.hstack(neg_list, np.zeros_like(neg_list))
        
        # 주변 화소 중에서 영역 평균 명암과 가장 근접한 화소를 영역으로 추가      
        dist = abs(neg_list[0:neg_pos, 2] - reg_mean) # 평균 명암과 거리계산 
        index = np.argmin(dist) # 거리가 최소인 화소와 
        pixdist = dist[index] # 그 화소값 
        reg_size = reg_size + 1 # 영역 사이즈 증가 
        
        # 새로운 영역의 평균 명암 계산 
        reg_mean = (reg_mean*reg_size + neg_list[index, 2])/(reg_size + 1)
        
        # x, y 좌표 저장
        x = int(neg_list[index, 0]) # 추가된 화소의 좌표를 현재점으로 
        y = int(neg_list[index, 1]) # 추가된 화소의 좌표를 현재점으로
        if pixdist < reg_maxdist:
            J[y, x] = 2 # 현재 화소를 새로운 영역으로 추가 
    
        # 주변화소 리스트에서 화소를 제거
        neg_list[index, :] = neg_list[neg_pos, :] # 영역으로 추가된 화소를 제거하고 그 공간에 새로 추가된 주변 후보 화소 정보를 할당 
        neg_pos = neg_pos - 1 # 주변 화소 접근 인덱스 감소 
        
    J = J > 1
    return J

if __name__ == "__main__":
    import matplotlib.pyplot as plt 
    
    OBJECT = 255
    BACK = 0
    
    I = np.zeros((10, 10), np.uint8)
    I[1:-1, 1:-3] = OBJECT 
    I[4:6, 3:5] = BACK  
    I = np.hstack((I, I))
    
    x, y = 5, 5
    reg_maxdist = 0.2
    
    # region growing 
    JJ = regiongrowing(I, x, y, reg_maxdist)
    
    plt.imshow(JJ)
    plt.show()