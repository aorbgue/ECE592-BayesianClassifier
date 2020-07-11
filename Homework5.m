%-------
% classification.m
% This script goes through examples in Hastie et al.'s book.
% Binary classification is performed in two ways: linear regression
% and nearest neighbors
% Dror Baron, 8.30.2016
%-------
clear % often useful to clean up the work space from old variables
%% Parameters
num_clusters=5; % number of components (clusters) in mixture model
N=200; % total number of samples of training data
grid=-3:0.1:3; % test data grid for each dimension
num_neighbors=20; % number of neighbors used in nearest neighbors
%% Mixture models
Gmean=randn(2,num_clusters); % locations of centers of clusters for green class
Rmean=randn(2,num_clusters); % -"- red class
%% Training data
samples=zeros(2,N); % locations of samples in 2 dimensions
cluster_id=zeros(2,N); % Created to identify the cluster id 
class_samples=zeros(N,1); % class of each one (green or red)
cluster_variance=0.1; % variance of each cluster around its mean
for n=1:N/2
    Gcluster=ceil(rand(1)*num_clusters); % select green cluster
    Rcluster=ceil(rand(1)*num_clusters); % -"- red
    cluster_id(:,n) = Gcluster; % Green ids from [1,num_clusters]
    cluster_id(:,n+N/2) = Rcluster; % Red ids from [1,num_clusters]
    samples(:,n)=Gmean(:,Gcluster)+sqrt(cluster_variance)*randn(2,1); % generate green sample
    samples(:,n+N/2)=Rmean(:,Rcluster)+sqrt(cluster_variance)*randn(2,1); % -"- red
    class_samples(n)=1; % green
    class_samples(n+N/2)=0; % red
end
%% Test data - basically a 2-D grid
test_samples=zeros(2,length(grid)^2); % locations of test samples
for n1=1:length(grid)
    for n2=1:length(grid)
        test_samples(1,n1+length(grid)*(n2-1))=grid(n1); % first coordinate
        test_samples(2,n1+length(grid)*(n2-1))=grid(n2); % second 
    end
end
%% Nearest Neighbors Classifier
test_NN=zeros(length(grid)^2,1); % classification results on test data
for n1=1:length(grid)
    for n2=1:length(grid)
        distances=(grid(n1)-samples(1,:)).^2+(grid(n2)-samples(2,:)).^2; % distances to training samples
        [distances_sort,distances_index]=sort(distances);
        neighbors=distances_index(1:num_neighbors);
        class_predicted=(sum(class_samples(neighbors))/num_neighbors>0.5); % NN classifier
        test_NN(n1+length(grid)*(n2-1))=class_predicted; % store classification
    end
end
%% PLOT NN
% identify location indices (in test grid) that are red and green
r_locations=find(test_NN==0);
g_locations=find(test_NN==1);
% NN plot
    figure(1),plot(samples(1,1:N/2),samples(2,1:N/2),'b*',... % green training samples
    samples(1,N/2+1:N),samples(2,N/2+1:N),'ro',... % red training 
    test_samples(1,g_locations),test_samples(2,g_locations),'b.',... % green test
    test_samples(1,r_locations),test_samples(2,r_locations),'r.'),... % red
title('Nearest Neighbors') 
axis([-3 3 -3 3]); % boundaries for figure aligned with grid
%% LDA (Section to solve 1c)
G_samples = samples(:,1:100);
R_samples = samples(:,101:200);
G_id = cluster_id(:,1:100);
R_id = cluster_id(:,101:200);

% Calculating Mean of the Classes
mean_G = zeros(2,num_clusters);
mean_R = zeros(2,num_clusters);
std_G = zeros(1,num_clusters);
std_R = zeros(1,num_clusters);

% Get means and variances from data
for i = 1:num_clusters
    meanG = G_samples(G_id==i);
    mean_G(:,i) = mean(reshape(meanG,[2,length(meanG)/2]),2);
    meanR = R_samples(R_id==i);
    mean_R(:,i) = mean(reshape(meanR,[2,length(meanR)/2]),2);
    stdG = G_samples(G_id==i);
    std_G_aux = var(reshape(stdG,[2,length(stdG)/2]),0,2);
    std_G(:,i) = mean(std_G_aux); % It is assumed that variance is the same for both dimensions
    stdR = R_samples(R_id==i); 
    std_R_aux = var(reshape(stdR,[2,length(stdR)/2]),0,2);
    std_R(:,i) = mean(std_R_aux); % It is assumed that variance is the same for both dimensions
end

% Class calculation in Test Data
Means = [mean_G,mean_R]; % New means calculated from data
Vars = mean([std_G,std_R]); % New variances calculated from data (LDA assumes same variance for all clusters)
test_LDA=zeros(length(grid)^2,1); % classification results on test data
for n1=1:length(grid)
    x1 = grid(n1);
    for n2=1:length(grid)
        x2 = grid(n2);
        p_x = exp(-0.5*(sum(([x1;x2]-Means).^2)./(Vars.^2))); % P(Class|X)
        class = p_x/sum(p_x); % Normalized P(Class|X)
        [max_value,class_predicted] = max(class); % Maximun[P(Class|X)] predicts class.
        test_LDA(n1+length(grid)*(n2-1))=class_predicted; % store classification
    end
end
%% PLOT LDA
% identify location indices (in test grid) that are red and green
r_locations=find(test_LDA>=num_clusters+1); % Subclasses [num_clusters+1,2*num_clusters]
g_locations=find(test_LDA<=num_clusters); % Subclasses [1,num_clusters]
% LDA plot
    figure(2),plot(samples(1,1:N/2),samples(2,1:N/2),'b*',... % green training samples
    samples(1,N/2+1:N),samples(2,N/2+1:N),'ro',... % red training 
    test_samples(1,g_locations),test_samples(2,g_locations),'b.',... % green test
    test_samples(1,r_locations),test_samples(2,r_locations),'r.'),... % red
title('Gaussian LDA')
axis([-3 3 -3 3]); % boundaries for figure aligned with grid
%% QDA (Section to solve 1d)
G_samples = samples(:,1:100);
R_samples = samples(:,101:200);
G_id = cluster_id(:,1:100);
R_id = cluster_id(:,101:200);

% Calculating Mean of the Classes
mean_G = zeros(2,num_clusters);
mean_R = zeros(2,num_clusters);
std_G = zeros(2,num_clusters);
std_R = zeros(2,num_clusters);

% Get means and variances from data
for i = 1:num_clusters
    meanG = G_samples(G_id==i);
    mean_G(:,i) = mean(reshape(meanG,[2,length(meanG)/2]),2);
    meanR = R_samples(R_id==i);
    mean_R(:,i) = mean(reshape(meanR,[2,length(meanR)/2]),2);
    stdG = G_samples(G_id==i);
    std_G(:,i) = var(reshape(stdG,[2,length(stdG)/2]),0,2);
    stdR = R_samples(R_id==i);
    std_R(:,i) = var(reshape(stdR,[2,length(stdR)/2]),0,2);
end

% Class calculation in Test Data
Means = [mean_G,mean_R]; % New means calculated from data
Vars = [std_G,std_R]; % New variances calculated from data
test_QDA=zeros(length(grid)^2,1); % classification results on test data
for n1=1:length(grid)
    x1 = grid(n1);
    for n2=1:length(grid)
        x2 = grid(n2);
        r = ([x1;x2]-Means).^2;
        px = r(1,:)./(Vars(1,:).^2);
        py = r(2,:)./(Vars(2,:).^2);
        p_x = exp(-0.5*(px+py)); % P(Class|X)
        class = p_x/sum(p_x); % Normalized P(Class|X)
        [max_value,class_predicted] = max(class); % Maximun[P(Class|X)] predicts class.
        test_QDA(n1+length(grid)*(n2-1))=class_predicted; % store classification
    end
end
%% PLOT QDA
% identify location indices (in test grid) that are red and green
r_locations=find(test_QDA>=num_clusters+1);
g_locations=find(test_QDA<=num_clusters);
% LDA plot
    figure(3),plot(samples(1,1:N/2),samples(2,1:N/2),'b*',... % green training samples
    samples(1,N/2+1:N),samples(2,N/2+1:N),'ro',... % red training 
    test_samples(1,g_locations),test_samples(2,g_locations),'b.',... % green test
    test_samples(1,r_locations),test_samples(2,r_locations),'r.'),... % red
title('Gaussian QDA')
axis([-3 3 -3 3]); % boundaries for figure aligned with grid