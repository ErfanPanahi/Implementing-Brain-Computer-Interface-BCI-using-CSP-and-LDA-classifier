%% Homework #7-8
% Amir Hossein Birjandi 810198367 
% Sogol Goodarzi 810198467
% Fatemeh Salehi 810198423
% Erfan Panahi 810198369

clc
clear
close all
%% Ptices of the brain
p1 = [2 4 6 7 11 33 35 39];
p2 = [1 3 5 8 36 37];
p3 = [9 10 14 34 38 43];
p4 = [15 16 20 40 44 48 49];
p5 = [12 13 17 21 22 41 45 46 50];
p6 = [18 19 23 42 47 51 52];
p7 = [24 25 29 31 53 57 61 63];
p8 = [26 32 54 55 58 60];
p9 = [27 28 30 56 59 62];
%% Data and Classes.
Data = load('subj_1.mat');
elbow = Data.data{1}; % Class 1
finger = Data.data{2}; % Class 2
foot = Data.data{3}; % Class 3
nothing = Data.data{4}; % Class 4
elbow(end,:,:) = [];
finger(end,:,:) = [];
foot(end,:,:) = [];
nothing(end,:,:) = [];
%----------
P = [p2 p4 p6 p7 p8 p9];
elbow = elbow(P,:,:);
finger = finger(P,:,:);
foot = foot(P,:,:);
nothing = nothing(P,:,:);
%----------
elbow = elbow - mean(elbow,2);
finger = finger - mean(finger,2);
foot = foot - mean(foot,2);
nothing = nothing - mean(nothing,2);
%% Filter
bf1 = [20,40]; %Hz
elbow1_fil = Filter(elbow,bf1);
finger1_fil = Filter(finger,bf1);
foot1_fil = Filter(foot,bf1);
nothing1_fil = Filter(nothing,bf1);
bf2 = [2,7]; %Hz
elbow2_fil = Filter(elbow,bf2);
finger2_fil = Filter(finger,bf2);
foot2_fil = Filter(foot,bf2);
bf3 = [1/3,3]; %Hz
elbow3_fil = Filter(elbow,bf3);
finger3_fil = Filter(finger,bf3);

%% Leave one out
dims = [size(elbow,3);size(finger,3);size(foot,3);size(nothing,3)];
confusion = zeros(4);
confusion_train = zeros(4);
tic
for i = 1:50
    leave = [randi([1,dims(1)]);randi([1,dims(2)]);randi([1,dims(3)]);randi([1,dims(4)])];
    
    %-------- test data
    testC1 = elbow1_fil(:,:,leave(1));
    testC2 = finger1_fil(:,:,leave(2));
    testC3 = foot1_fil(:,:,leave(3));   
    testC4 = nothing1_fil(:,:,leave(4));
    
    %-------- train data
    C4 = nothing1_fil; 
    C4(:,:,leave(4)) = [];
    C3 = foot1_fil; 
    C3(:,:,leave(3)) = [];
    C2 = finger1_fil; 
    C2(:,:,leave(2)) = [];
    C1 = elbow1_fil; 
    C1(:,:,leave(1)) = [];
    
    %-------- find correlations
    R_elbow = R(C1);
    R_finger = R(C2);
    R_foot = R(C3);
    R_nothing = R(C4);
    
    %-------- classifying "Nothing"
    R_others1 = (R_elbow + R_foot + R_finger) / 3;
    W_csp1 = findWcsp(R_nothing,R_others1,7);
    FeatureC4 = FeatureExt(C4,W_csp1);
    FeatureC3 = FeatureExt(C3,W_csp1);
    FeatureC2 = FeatureExt(C2,W_csp1);
    FeatureC1 = FeatureExt(C1,W_csp1);
    
    mu_c4 = mean(FeatureC4,2);
    mu_c3 = mean(FeatureC3,2);
    mu_c2 = mean(FeatureC2,2);
    mu_c1 = mean(FeatureC1,2);
    mu1 = mu_c4;
    mu2 = (mu_c1 + mu_c2 + mu_c3)/3;
    
    SigmaC4 = SIGMAmat(FeatureC4,mu1,size(C4,3));
    SigmaC3 = SIGMAmat(FeatureC3,mu_c3,size(C3,3));
    SigmaC2 = SIGMAmat(FeatureC2,mu_c2,size(C2,3));
    SigmaC1 = SIGMAmat(FeatureC1,mu_c1,size(C1,3));
    
    Sigma1 = SigmaC4;
    Sigma2 = (SigmaC3 + SigmaC2 + SigmaC1)/3;
    
    W_lda1 = findWlda(mu1,mu2,Sigma1,Sigma2);
    mu1_1 = W_lda1.' * mu1;
    mu1_2 = W_lda1.' * mu2;
    thr1 = (mu1_1 + mu1_2)/2; % Threshold for "Nothing" 
    
    temp = zeros(4);
    trainC4 = zeros(size(C4,3),4);
    trainC3 = zeros(size(C3,3),4);
    trainC2 = zeros(size(C2,3),4);
    trainC1 = zeros(size(C1,3),4);
    
    U = [1;1;1;1];
    U_C4 = ones(size(C4,3),1);
    U_C3 = ones(size(C3,3),1);
    U_C2 = ones(size(C2,3),1);
    U_C1 = ones(size(C1,3),1);

    temp(:,4) = findLABEL(W_csp1,W_lda1,testC1,testC2,testC3,testC4,mu1_1,thr1,U);
    
    trainC4(:,4) = findLABEL_Train(W_csp1,W_lda1,C4,mu1_1,thr1,U_C4);
    trainC3(:,4) = findLABEL_Train(W_csp1,W_lda1,C3,mu1_1,thr1,U_C3);
    trainC2(:,4) = findLABEL_Train(W_csp1,W_lda1,C2,mu1_1,thr1,U_C2);
    trainC1(:,4) = findLABEL_Train(W_csp1,W_lda1,C1,mu1_1,thr1,U_C1);

    %-------- test data
    testC1 = elbow2_fil(:,:,leave(1));
    testC2 = finger2_fil(:,:,leave(2));
    testC3 = foot2_fil(:,:,leave(3));   
    testC4 = nothing2_fil(:,:,leave(4));
    
    %-------- train data
    C3 = foot2_fil; 
    C3(:,:,leave(3)) = [];
    C2 = finger2_fil; 
    C2(:,:,leave(2)) = [];
    C1 = elbow2_fil; 
    C1(:,:,leave(1)) = [];
    
    %-------- find correlations
    R_elbow = R(C1);
    R_finger = R(C2);
    R_foot = R(C3);
    R_nothing = R(C4);
    
    %-------- classifying "Foot"
    R_others2 = (R_elbow + R_finger) / 2;
    W_csp2 = findWcsp(R_foot,R_others2,3);
    FeatureC3 = FeatureExt(C3,W_csp2);
    FeatureC2 = FeatureExt(C2,W_csp2);
    FeatureC1 = FeatureExt(C1,W_csp2);
    
    mu_c3 = mean(FeatureC3,2);
    mu_c2 = mean(FeatureC2,2);
    mu_c1 = mean(FeatureC1,2);
    mu1 = mu_c3;
    mu2 = (mu_c1 + mu_c2)/2;
    
    SigmaC3 = SIGMAmat(FeatureC3,mu1,size(C3,3));
    SigmaC2 = SIGMAmat(FeatureC2,mu_c2,size(C2,3));
    SigmaC1 = SIGMAmat(FeatureC1,mu_c1,size(C1,3));
    
    Sigma1 = SigmaC3;
    Sigma2 = (SigmaC2 + SigmaC1)/2;
    
    W_lda2 = findWlda(mu1,mu2,Sigma1,Sigma2);
    mu2_1 = W_lda2.' * mu1;
    mu2_2 = W_lda2.' * mu2;
    thr2 = (mu2_1 + mu2_2)/2; % Threshold for "Foot"
    
    U = U .* ~temp(:,4);
    U_C4 = U_C4 .* ~trainC4(:,4);
    U_C3 = U_C3 .* ~trainC3(:,4);
    U_C2 = U_C2 .* ~trainC2(:,4);
    U_C1 = U_C1 .* ~trainC1(:,4);

    temp(:,3) = findLABEL(W_csp2,W_lda2,testC1,testC2,testC3,testC4,mu2_1,thr2,U);
    trainC4(:,3) = findLABEL_Train(W_csp2,W_lda2,C4,mu2_1,thr2,U_C4);
    trainC3(:,3) = findLABEL_Train(W_csp2,W_lda2,C3,mu2_1,thr2,U_C3);
    trainC2(:,3) = findLABEL_Train(W_csp2,W_lda2,C2,mu2_1,thr2,U_C2);
    trainC1(:,3) = findLABEL_Train(W_csp2,W_lda2,C1,mu2_1,thr2,U_C1);
    
    %-------- test data
    testC1 = elbow3_fil(:,:,leave(1));
    testC2 = finger3_fil(:,:,leave(2));
    testC3 = foot3_fil(:,:,leave(3));   
    testC4 = nothing3_fil(:,:,leave(4));
    
    %-------- train data
    C2 = finger3_fil; 
    C2(:,:,leave(2)) = [];
    C1 = elbow3_fil; 
    C1(:,:,leave(1)) = [];
    
    %-------- find correlations
    R_elbow = R(C1);
    R_finger = R(C2);
    R_foot = R(C3);
    R_nothing = R(C4);
    
    %-------- classifying "Finger"
    W_csp3 = findWcsp(R_elbow,R_finger,10);
    FeatureC2 = FeatureExt(C2,W_csp3);
    FeatureC1 = FeatureExt(C1,W_csp3);
    
    mu1 = mean(FeatureC2,2);
    mu2 = mean(FeatureC1,2);
    
    Sigma1 = SIGMAmat(FeatureC2,mu1,size(C2,3));
    Sigma2 = SIGMAmat(FeatureC1,mu2,size(C1,3));
    
    W_lda3 = findWlda(mu1,mu2,Sigma1,Sigma2);
    mu3_1 = W_lda3.' * mu1;
    mu3_2 = W_lda3.' * mu2;
    thr3 = (mu3_1 + mu3_2)/2; % Threshold for "Finger"
    
    U = U .* ~temp(:,3);
    U_C4 = U_C4 .* ~trainC4(:,3);
    U_C3 = U_C3 .* ~trainC3(:,3);
    U_C2 = U_C2 .* ~trainC2(:,3);
    U_C1 = U_C1 .* ~trainC1(:,3);

    temp(:,2) = findLABEL(W_csp3,W_lda3,testC1,testC2,testC3,testC4,mu3_1,thr3,U);
	trainC4(:,2) = findLABEL_Train(W_csp3,W_lda3,C4,mu3_1,thr3,U_C4);
	trainC3(:,2) = findLABEL_Train(W_csp3,W_lda3,C3,mu3_1,thr3,U_C3);
	trainC2(:,2) = findLABEL_Train(W_csp3,W_lda3,C2,mu3_1,thr3,U_C2);
	trainC1(:,2) = findLABEL_Train(W_csp3,W_lda3,C1,mu3_1,thr3,U_C1);

    temp(:,1) = 1 - (temp(:,2) + temp(:,3) + temp(:,4));
    trainC4(:,1) = 1 - (trainC4(:,2) + trainC4(:,3) + trainC4(:,4));
    trainC3(:,1) = 1 - (trainC3(:,2) + trainC3(:,3) + trainC3(:,4));
    trainC2(:,1) = 1 - (trainC2(:,2) + trainC2(:,3) + trainC2(:,4));
    trainC1(:,1) = 1 - (trainC1(:,2) + trainC1(:,3) + trainC1(:,4));
    
    trainC4 = sum(trainC4);
    trainC3 = sum(trainC3);
    trainC2 = sum(trainC2);
    trainC1 = sum(trainC1);
    
    confusion = confusion + temp;
    confusion_train = confusion_train + [trainC1;trainC2;trainC3;trainC4];
end
toc
confusion/50
[confusion_train(1,:)/sum(confusion_train(1,:))
   confusion_train(2,:)/sum(confusion_train(2,:)) 
   confusion_train(3,:)/sum(confusion_train(3,:))
   confusion_train(4,:)/sum(confusion_train(4,:))]
%% Functions.
function [x_fil] = Filter(x,bf)
    x_fil = zeros(size(x));
    L0 = size(x,2)/2;
    Xfil = zeros(1,2*L0);
    for i = 1:size(x,3)
        for j = 1:size(x,1)
            F_x = fftshift(fft(x(j,:,i)));
            Xfil(L0-bf(2)*3+1:L0-bf(1)*3) = F_x(L0-bf(2)*3+1:L0-bf(1)*3);
            Xfil(L0+bf(1)*3+1:L0+bf(2)*3) = F_x(L0+bf(1)*3+1:L0+bf(2)*3);
            x_fil(j,:,i) = ifft(ifftshift(Xfil));
        end
    end
end

function [Rx] = R(x)
    L = size(x,1);
    Rx = zeros(L);
    t = size(x,3);
    for j = 1:t
        Rx = Rx + x(:,:,j)*(x(:,:,j))';
    end
    Rx = Rx/63;
end

function [W_csp] = findWcsp(R1,R2,L)
    [W_CSP,LANDA] = eig(R1,R2);
    [~,pos] = sort(diag(LANDA),'descend');
    W_CSP = W_CSP(:,pos);
    W_CSP =  W_CSP ./ sqrt(sum(W_CSP.^2));
    W_csp = W_CSP(:,[1:L,end-(L-1):end]);
end

function [Feature] = FeatureExt(X,Wcsp)
    N = size(Wcsp,2); % number of features
    Feature = zeros(N,size(X,3));
    for j = 1:size(X,3)
        x = X(:,:,j);
        WtX = Wcsp.' * x;
        Feature(:,j) = (var(WtX.')).';
    end
end

function [Sigma] = SIGMAmat(Feature,mu,L)
    N = size(Feature,1);
    Sigma = zeros(N);
    for j = 1:L
        Sigma = Sigma + ((Feature(:,j)-mu)*(Feature(:,j)-mu).');
    end
    Sigma = Sigma / L;
end

function [W_lda] = findWlda(mu1,mu2,Sigma1,Sigma2)
    A = (mu1-mu2)*(mu1-mu2).';
    B = Sigma1 + Sigma2;
    [W_LDA,LANDA] = eig(A,B);
    [~,pos] = sort(diag(LANDA),'descend');
    W_LDA = W_LDA(:,pos);
    W_lda = W_LDA(:,1);  
end

function [label] = findLABEL(W_csp,W_lda,testC1,testC2,testC3,testC4,mu_1,thr,U)
    FeatureTestC1 = var((W_csp.' * testC1).').';
    FeatureTestC2 = var((W_csp.' * testC2).').';
    FeatureTestC3 = var((W_csp.' * testC3).').';
    FeatureTestC4 = var((W_csp.' * testC4).').';
    mu_test1 = W_lda.' * FeatureTestC1;
    mu_test2 = W_lda.' * FeatureTestC2;
    mu_test3 = W_lda.' * FeatureTestC3;
    mu_test4 = W_lda.' * FeatureTestC4;
    mu_TEST = [mu_test1,mu_test2,mu_test3,mu_test4];
    lower_thr = mu_TEST < thr;
    if (mu_1 < thr)
        label = lower_thr.' .* U;
    else
        label = ~lower_thr.' .* U;
    end
end
function [label] = findLABEL_Train(W_csp,W_lda,train,mu_1,thr,U)
    FeatureTrain = zeros(size(W_csp,2),size(train,3));
    for i = 1:size(train,3)
        FeatureTrain(:,i) = var((W_csp.' * train(:,:,i)).').';
    end
    mu_train = W_lda.' * FeatureTrain;
    lower_thr = mu_train' < thr;
    if (mu_1 < thr)
        label = lower_thr .* U;
    else
        label = ~lower_thr .* U;
    end
end