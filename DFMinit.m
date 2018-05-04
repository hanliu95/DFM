function [w0,w,userV,itemV,V3,userZ,itemZ,Z3] = DFMinit( userX,itemX,fX, y,F, k, alpha, beta, option )
X = [userX,itemX,fX];
[userId,~] = find(userX');
[itemId,~] = find(itemX');
unum = size(userX,2);%number of users
inum = size(itemX,2);%number of items
fnum = size(fX,2);%number of item features
w0 = rand;
w = rand(unum+inum+fnum,1);
userV = rand(k,unum);
itemV = rand(k,inum);
V3 = rand(k,fnum);
userZ = UpdateSVD(userV);
itemZ = UpdateSVD(itemV);
Z3 = UpdateSVD(V3);

if isfield(option,'maxItr')
    maxItr = option.maxItr;
else
    maxItr = 50;
end
tol = 1e-5;
userIDX = (userX~=0);
itemIDX = (itemX~=0);
%FIDX = (F~=0);

disp('Starting DFMinit...');
[loss,obj] = DFMinitobj(userX,itemX,fX,y,F,k,alpha,beta,w0,w,userV,itemV,V3,userZ,itemZ,Z3);
disp(['loss value = ',num2str(loss),' obj value = ',num2str(obj)]);

converge = false;
it = 1;
p = ScaleScore(y,2*k);
while ~converge
    %optimize userV
    userV0 = userV;
    itemV0 = itemV;
    V30 = V3;
    p1 = p - w0 - X*w;
    F1 = V3*F';
    parfor u = 1:unum 
       D = itemV(:,itemId(userIDX(:,u))) + F1(:,itemId(userIDX(:,u)));
       pu = p1(userIDX(:,u));
       Q = D*D' + beta*length(pu)*eye(k);
       L = D*pu + beta*userZ(:,u);
       userV(:,u) = pinv(Q)*L;
    end
    %optimize itemV
    parfor i = 1:inum
        D = userV(:,userId(itemIDX(:,i)));
        pi = p1(itemIDX(:,i)) - D'*F1(:,i);
        Q = D*D' + beta*length(pi)*eye(k);
        L = D*pi + beta*itemZ(:,i);
        itemV(:,i) = pinv(Q)*L;
    end
    
    %optimize V3
    for j = 1:fnum 
        a = find(F(:,j));% a is the array storages item having feature_j
        Q = zeros(k,k);
        L = zeros(k,1);
        parfor t = 1:length(a)
           b = a(t);%the itemId 
           D = userV(:,userId(itemIDX(:,b)));% * F(b,j);
           c = F(b,:); c(j) = 0;
           pt = p1(itemIDX(:,b)) - D'*itemV(:,b) - D'*V3*c';
           Q = Q + D*D'*(F(b,j)^2);
           L = L + D*pt*F(b,j);
        end
        Q = Q + beta*length(a)*unum/2*eye(k);
        L = L + beta*Z3(:,j);
        V3(:,j) = Q\L;
    end
    
    %optimize userZ and itemZ 
    %disp('...');
    userZ = UpdateSVD(userV);
    itemZ = UpdateSVD(itemV);
    Z3 = UpdateSVD(V3);
    
    %optimize w0 and w
    p2 = p;
    F2 = V3*F';
    parfor m = 1:length(p)
        p2(m) = p2(m) - userV(:,userId(m))'*( itemV(:,itemId(m)) + F2(:,itemId(m)) );
    end
    %
    D = [ones(length(y),1),userX,itemX,fX];
    Q = D'*D + alpha*speye(unum+inum+fnum+1);
    L = D'*p2;
    w = pcg(Q,L);
    w0 = w(1); w(1) = [];
    
    disp(['DFMinit iteration: ',int2str(it)]);
    [loss,obj] = DFMinitobj(userX,itemX,fX,y,F,k,alpha,beta,w0,w,userV,itemV,V3,userZ,itemZ,Z3);
    disp(['loss value = ',num2str(loss),' obj value = ',num2str(obj)]); 
    
    if it >= maxItr || max([norm(userV0-userV,'fro'), norm(itemV0-itemV,'fro'), norm(V30-V3,'fro')])<max([unum,inum,fnum])*tol
      converge = true; 
    end
    it = it+1;
    
end

end

function [loss,obj] = DFMinitobj(userX,itemX,fX,y,F,k,alpha,beta,w0,w,userV,itemV,V3,userZ,itemZ,Z3)
X = [userX,itemX,fX];
V = [userV,itemV,V3];
Z = [userZ,itemZ,Z3];
[userId,~] = find(userX');
[itemId,~] = find(itemX');
p = ScaleScore(y,2*k);
p = p - w0 - X*w;
F2 = V3*F';
parfor m = 1:length(y)
    p(m) = p(m) - userV(:,userId(m))'*( itemV(:,itemId(m)) + F2(:,itemId(m)) );
end
loss = p'*p;
no = norm(V,'fro');
obj = loss + alpha*w'*w + beta*no.^2 + beta*trace(V*Z');
end
