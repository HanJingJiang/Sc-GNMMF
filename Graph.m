clc,clear all
% ———————————求相似矩阵的稀疏图————————————————
load('Data');
LL=gene; 
DD=cell; 
L=LL-diag(diag(LL));
D=DD-diag(diag(DD));
[r1,c1]=size(L);
[r2,c2]=size(D);
P1=5;
P2=5;
PNN_L = zeros(r1, c1);
graph_L = zeros(r1, c1);
[sort_L,idx]=sort(L,2,'descend');
 for i = 1 : r1
        PNN_L(i,idx(i,1:P1))=sort_L(i,1:P1);
    end    
     for i = 1 : r1
        idx_i=find(PNN_L(i,:));
        for j = 1 : r1           
            idx_j=find(PNN_L(j,:));
            if ismember(j,idx_i) & ismember(i,idx_j) %&& isequal(Clu_mat(i,j),1)               
                graph_L(i,j)=1;
            elseif ~ismember(j,idx_i) & ~ismember(i,idx_j) %&& ~isequal(Clu_mat(i,j),1)  
                graph_L(i,j)=0;
            else
                graph_L(i,j)=0.5;               
            end       
        end
     end

PNN_D = zeros(r2, c2);
graph_D = zeros(r2, c2);
[sort_D,idx]=sort(D,2,'descend');
 for i = 1 : r2
        PNN_D(i,idx(i,1:P2))=sort_D(i,1:P2);
    end    
     for i = 1 : r2
        idx_i=find(PNN_D(i,:));
        for j = 1 : r2           
            idx_j=find(PNN_D(j,:));
            if ismember(j,idx_i) & ismember(i,idx_j) %&& isequal(Clu_mat(i,j),1)               
                graph_D(i,j)=1;
            elseif ~ismember(j,idx_i) & ~ismember(i,idx_j) %&& ~isequal(Clu_mat(i,j),1)  
                graph_D(i,j)=0;
            else
                graph_D(i,j)=0.5;               
            end       
        end
     end
     
     
% ———————————求稀疏化相似矩阵———————————————
S_L=LL.*graph_L;      % 相似矩阵的稀疏化（P近邻法）
S_D=DD.*graph_D;      % 相似矩阵的稀疏化（P近邻法）
save('S_L_D','S_L','S_D');
