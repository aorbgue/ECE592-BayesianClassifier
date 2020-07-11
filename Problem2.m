%-------
% classification.m
% This script goes through examples in Hastie et al.'s book.
% Binary classification is performed in two ways: linear regression
% and nearest neighbors
% Dror Baron, 8.30.2016
%-------

clear % often useful to clean up the work space from old variables
%% Parameters
num_clusters=3; % number of components (clusters) in mixture model
N=200; % total number of samples of training data
grid=-3:0.1:3; % test data grid for each dimension
num_neighbors=10; % number of neighbors used in nearest neighbors
%% Mixture Models
Gmean=randn(2,num_clusters); % locations of centers of clusters for green class
Rmean=randn(2,num_clusters); % -"- red class
%% Training data
samples=zeros(2,N); % locations of samples in 2 dimensions
class_samples=zeros(N,1); % class of each one (green or red)
cluster_variance=0.1; % variance of each cluster around its mean
for n=1:N/2
    Gcluster=ceil(rand(1)*num_clusters); % select green cluster
    Rcluster=ceil(rand(1)*num_clusters); % -"- red
    samples(:,n)=Gmean(:,Gcluster)+sqrt(cluster_variance)*randn(2,1); % generate green sample
    samples(:,n+N/2)=Rmean(:,Rcluster)+sqrt(cluster_variance)*randn(2,1); % -"- red
    class_samples(n)=1; % green
    class_samples(n+N/2)=0; % red
end
%% Test Data - basically a 2-D grid
test_samples=zeros(2,length(grid)^2); % locations of test samples
for n1=1:length(grid)
    for n2=1:length(grid)
        test_samples(1,n1+length(grid)*(n2-1))=grid(n1); % first coordinate
        test_samples(2,n1+length(grid)*(n2-1))=grid(n2); % second 
    end
end
%% Nearest Neighbors (Clasical Approach)
test_NN=zeros(length(grid)^2,1); % classification results on test data
for n1=1:length(grid)
    for n2=1:length(grid)
        distances=(grid(n1)-samples(1,:)).^2+(grid(n2)-samples(2,:)).^2; % distances to training samples
        [distances_sort,distances_index]=sort(distances);
        neighbors=distances_index(1:num_neighbors);
        class_predicted=(sum(class_samples(neighbors))/num_neighbors>0.5); % NN classifier: 0 = RED, 1 = BLUE
        test_NN(n1+length(grid)*(n2-1))=class_predicted; % store classification
    end
end
%% PLOT
% identify location indices (in test grid) that are red and green
r_locations=find(test_NN==0);
g_locations=find(test_NN==1);
% NN plot
figure(1),plot(samples(1,1:N/2),samples(2,1:N/2),'b*',... % green training samples
    samples(1,N/2+1:N),samples(2,N/2+1:N),'ro',... % red training 
    test_samples(1,g_locations),test_samples(2,g_locations),'b.',... % green test
    test_samples(1,r_locations),test_samples(2,r_locations),'r.') % red
title('NN Original')
axis([-3 3 -3 3]); % boundaries for figure aligned with grid
%% Nearest Neighbors (Problem 2) - MODIFICATION
test_NN_2=zeros(length(grid)^2,1); % classification results on test data
for n1=1:length(grid)
    for n2=1:length(grid)
        distances=(grid(n1)-samples(1,:)).^2+(grid(n2)-samples(2,:)).^2; % distances to training samples
        [distances_sort,distances_index]=sort(distances);
        neighbors=distances_index(1:num_neighbors);
        NN_distances = 2*(num_neighbors:-1:1)./((num_neighbors)*(num_neighbors+1));
        Rsum = sum(NN_distances(class_samples(neighbors)==0)); % Sum K distances of all neighbors in Red Class
        Gsum = sum(NN_distances(class_samples(neighbors)==1)); % Sum K distances of all neighbors in Green Class
        class_predicted=Gsum>Rsum; % NN classifier: 0 = RED, 1 = BLUE
        test_NN_2(n1+length(grid)*(n2-1))=class_predicted; % store classification
    end
end
%% PLOT NN Modified
% identify location indices (in test grid) that are red and green
r_locations=find(test_NN_2==0);
g_locations=find(test_NN_2==1);
% NN plot
figure(2),plot(samples(1,1:N/2),samples(2,1:N/2),'b*',... % green training samples
    samples(1,N/2+1:N),samples(2,N/2+1:N),'ro',... % red training 
    test_samples(1,g_locations),test_samples(2,g_locations),'b.',... % green test
    test_samples(1,r_locations),test_samples(2,r_locations),'r.') % red
title('NN Modified')
axis([-3 3 -3 3]); % boundaries for figure aligned with grid
