function [ w0,w,userU,itemU,U3,userZ,itemZ,Z3 ] = DFM( userX,itemX, fX, y,F, k, alpha, beta, option )
%DFM 此处显示有关此函数的摘要
%   此处显示详细说明 DFM_v1
X = [userX,itemX,fX];
[userId,~] = find(userX');
[itemId,~] = find(itemX');
unum = size(userX,2);%number of users
inum = size(itemX,2);%number of items
fnum = size(fX,2);%number of features
userIDX = (userX~=0);
itemIDX = (itemX~=0);

if isfield(option,'maxItr')
    maxItr = option.maxItr;
else
    maxItr = 50;
end
maxItr2 = 5;

%[uNum,iNum] = size(R);
%IDX = (R~=0);
%IDXT = IDX';
%R = ScaleScore(R,k);
%RT = R';
lf = zeros(31,1);
of = zeros(31,1);

[w0,w,userV,itemV,V3,userZ,itemZ,Z3] = DFMinit(userX,itemX,fX, y,F, k, alpha, beta, option);
userU = sign(userV); userU(userU == 0) = 1;
itemU = sign(itemV); itemU(itemU == 0) = 1;
U3 = sign(V3); U3(U3 == 0) = 1;

disp('Starting DFM... ');
%[loss,obj] = DFMobj(userX,itemX,fX,y,F,k,alpha,beta,w0,w,userU,itemU,U3,userZ,itemZ,Z3);
%disp(['loss value = ',num2str(loss),' obj value = ',num2str(obj)]);

converge = false;
it = 1;
p = ScaleScore(y,2*k);
while ~converge
   userU0 = userU;
   itemU0 = itemU;
   U30 = U3;
   p1 = p - w0 - X*w; 
   F1 = U3*F';
   %optimize userU
   parfor u = 1:unum
      uu = userU(:,u);
      %iU = itemU(:,IDXT(:,u));
      %DCDmex(uu,iU*iU',iU*nonzeros(RT(:,u)),beta*userZ(:,u),maxItr2);
      D = itemU(:,itemId(userIDX(:,u))) + F1(:,itemId(userIDX(:,u)));
      pu = p1(userIDX(:,u));
      DCDmex(uu,D*D',D*pu,beta*userZ(:,u),maxItr2);
      userU(:,u) = uu;
   end
   %optimize itemU
   parfor i = 1:inum
      iu = itemU(:,i);
      %uU = userU(:,IDX(:,i));
      %DCDmex(iu,uU*uU',uU*nonzeros(R(:,i)),beta*itemZ(:,i),maxItr2);
      D = userU(:,userId(itemIDX(:,i)));
      pi = p1(itemIDX(:,i)) - D'*F1(:,i);
      DCDmex(iu,D*D',D*pi,beta*itemZ(:,i),maxItr2);
      itemU(:,i) = iu;
   end
   %optimize U3
   for j = 1:fnum %the index of features,amazon yelp = 20, movie = 18
       fu = U3(:,j);
       a = find(F(:,j));% a is the array storages item having feature_j
       %D1 = itemV(:,a);%the v of items with feature j
       Q = zeros(k,k);
       L = zeros(k,1);
       parfor t = 1:length(a)
          b = a(t);%the itemId 
          D = userU(:,userId(itemIDX(:,b)));% * F(b,j);
          c = F(b,:); c(j) = 0;% without feature j
          pt = p1(itemIDX(:,b)) - D'*itemU(:,b) - D'*U3*c';
          Q = Q + D*D'*(F(b,j)^2);
          L = L + D*pt*F(b,j);
       end
       %Q = Q + beta*length(a)*unum/2*eye(k);
       %L = L + beta*Z3(:,j);
       %V3(:,j) = Q\L;
       DCDmex(fu,Q,L,beta*U3(:,j),maxItr2);
       U3(:,j) = fu;
   end
   %optimize userZ and itemZ
   userZ = UpdateSVD(userU);
   itemZ = UpdateSVD(itemU);
   Z3 = UpdateSVD(U3);
   
   %optimize w
   p2 = p;
   F2 = U3*F';
   parfor m = 1:length(p2)
      p2(m) = p2(m) - userU(:,userId(m))'*( itemU(:,itemId(m)) + F2(:,itemId(m)) );
   end
   D = [ones(length(y),1),userX,itemX,fX];
   Q = D'*D + alpha*speye(unum+inum+fnum+1);
   L = D'*p2;
   w = pcg(Q,L);
   w0 = w(1); w(1) = [];
   
   disp(['DFM at bit ',int2str(k),' Iteration:',int2str(it)]);
   %[loss,obj] = DFMobj(userX, itemX, fX, y, F, k, alpha, beta, w0, w, userU, itemU, U3, userZ, itemZ, Z3);
   %disp(['loss value = ',num2str(loss),' obj value = ',num2str(obj)]);
   %lf(1+it)=loss;
   %of(1+it)=obj;
   
   if it >= maxItr%||(sum(sum(userU~=userU0))==0 && sum(sum(itemU~=itemU0))==0&&sum(sum(U3~=U30))==0) 
     converge = true; 
   end
   it = it+1;
   
end
%save fvalue2 lf of;

end

function [loss,obj] = DFMobj(userX,itemX,fX,y,F,k,alpha,beta,w0,w,userU,itemU,U3,userZ,itemZ,Z3)
X = [userX,itemX,fX];
U = [userU,itemU,U3];
Z = [userZ,itemZ,Z3];
[userId,~] = find(userX');
[itemId,~] = find(itemX');
p = ScaleScore(y,2*k); 
p = p - w0 - X*w;
F2 = U3*F';
parfor m = 1:length(y)
    p(m) = p(m) - userU(:,userId(m))'*( itemU(:,itemId(m)) + F2(:,itemId(m)) );
end
loss = p'*p;
obj = loss + alpha*(w'*w)  + beta*trace(U*Z');
end