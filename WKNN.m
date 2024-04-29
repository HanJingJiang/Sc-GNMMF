clc,clear all
load('Data') %LD为原始矩阵
D=cell; %cell的相似矩阵
L=gene;
oldLD = LD;
adLD = oldLD >0;
%（最佳参数：K=3,a=0.5，a1=a2=1）
%————————————————以下求KNN————————————————
L=L-diag(diag(L));
D=D-diag(diag(D));
[rL,cL]=size(L);
[rD,cD]=size(D);
K=5;  %参数K一般取1，2，3，4，5    
KNN_L = zeros(rL, cL);  %%for miRNA
[sort_L,idx]=sort(L,2,'descend');
 for i = 1 : rL
        KNN_L(i,idx(i,1:K))=sort_L(i,1:K);
 end  

KNN_D = zeros(rD, cD);  %%for disease
[sort_D,idx]=sort(D,2,'descend');
 for i = 1 : rD
        KNN_D(i,idx(i,1:K))=sort_D(i,1:K);
 end     

%————————————以下求新的权重矩阵——————————————-
[rows,cols]=size(adLD);  
y_l=zeros(rows,cols);  
y_d=zeros(rows,cols); 
a=0.5; %参数a=[0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9]

knn_network_l =KNN_L;  %for miRNA
for i = 1 : rows   
        w=zeros(1,K);
        [sort_l,idx_l]=sort(knn_network_l(i,:),2,'descend'); 
        sum_l=sum(sort_l(1,1:K));   
        for j = 1 : K
            w(1,j)=a^(j-1)*sort_l(1,j); 
            y_l(i,:) =  y_l(i,:)+ w(1,j)* LD(idx_l(1,j),:); 
        end                      
            y_l(i,:)=y_l(i,:)/sum_l;              
end

knn_network_d =KNN_D;  %for disease
for i = 1 : cols   
        w1=zeros(1,K);
        [sort_d,idx_d]=sort(knn_network_d(i,:),2,'descend'); 
        sum_d=sum(sort_d(1,1:K));   
        for j = 1 : K
            w1(1,j)=a^(j-1)*sort_d(1,j); 
          y_d(:,i) =  y_d(:,i)+ w1(1,j)* adLD(:,idx_d(1,j)); 
        end                      
            y_d(:,i)=y_d(:,i)/sum_d;              
end

% ———————————————求新的关联矩阵——————————————
for i = 1 : rows
     for j = 1 : cols
         LD_1(i,j)=max(LD(i,j),y_d(i,j));  
     end    
 end
LD_new = LD_1;
save('LD1.mat', 'LD_new')

 
