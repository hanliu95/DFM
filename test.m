%training set
%R = getfield(load('train_movie.mat','train_movie_Matrix'),'train_movie_Matrix');
R = getfield(load('train_amazon.mat','train_amazon_Matrix'),'train_amazon_Matrix');
%R = getfield(load('train_yelp.mat','train_yelp_Matrix'),'train_yelp_Matrix');


%form FMs model feature vectors from transaction data
[userId,itemId,y] = find(R);
num = length(y);
userX = sparse(1:num,userId,ones(num,1));
itemX = sparse(1:num,itemId,ones(num,1));

%DFM_2.0
%F = getfield(load('feature_movie.mat','feature_movie_Matrix'),'feature_movie_Matrix');
F = getfield(load('feature_amazon.mat','feature_amazon_Matrix'),'feature_amazon_Matrix');
%F = getfield(load('feature_yelp.mat','feature_yelp_Matrix'),'feature_yelp_Matrix');

if size(F,2) > 100
    h = sum(F);
    [~,top] = sort(h,'descend');
    F = F(:,top(1:100));
end

d = sum(F,2);
dIDX = (d == 0);
F = bsxfun(@rdivide,F,d);
F(dIDX,:) = 0;
fX = F(itemId,:);
%DFM_2.0

%define the dimensionality of the factorization
k = 64;
%define the tunning parameter
alpha = 0;
beta = 10;
%apply initialization
option.init = true;
option.debug = true;
%number of iterations
option.maxItr = 20;
[w0,w,userU,itemU,U3,userZ,itemZ,Z3] = DFM(userX,itemX,fX, y,F,k,alpha,beta,option);
%save yover128_401_p w0 w userU itemU U3;
%save DFM_2_8_movie_parameter w0 w userU itemU U3;
%save DFM_2_8_amazon_parameter w0 w userU itemU U3;
%save DFM_2_64_yelp_parameter w0 w userU itemU U3;

%evaluation measure NDCG

%test set
%S = getfield(load('test_movie.mat','test_movie_Matrix'),'test_movie_Matrix');
S = getfield(load('test_amazon.mat','test_amazon_Matrix'),'test_amazon_Matrix');
%S = getfield(load('test_yelp.mat','test_yelp_Matrix'),'test_yelp_Matrix');

[userId2,itemId2,y2] = find(S);
num2 = length(y2);
userX2 = sparse(1:num2,userId2,ones(1,num2));
itemX2 = sparse(1:num2,itemId2,ones(1,num2));
fX2 = F(itemId2,:);
%
ndcg = zeros(10,1);
for kvalue = 1:10
    [ndcgvalue] = NDCG(userX2,itemX2,fX2, y2, F, w0,w,userU,itemU,U3,kvalue);
    ndcg(kvalue) = ndcgvalue;
    disp(['The DFM ndcg@',int2str(kvalue),' is ',num2str(ndcgvalue)]);
end
%plot(ndcg);
%}
%ndcg10 = NDCG(userX2,itemX2,fX2, y2, F, w0,w,userU,itemU,U3,10);
%save a64 ndcg
%disp(ndcg10);
save ao_10 ndcg;