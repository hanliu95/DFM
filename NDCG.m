function [ ndcgvalue ] = NDCG( userX,itemX,fX, y,F, w0,w,userU, itemU, U3, kvalue )
%compute the ndcg@kvalue of discrete factorization machines
[itemId,~] = find(itemX');
userIDX = (userX~=0);
unum = size(userX,2);
ndcg = zeros(unum,1);

invalid = 0;
h = [userX,itemX,fX]*w;
F2 = U3*F';
parfor u = 1:unum
   %compute the dcg of the query user u
   itemlist = itemId(userIDX(:,u));
   target = y(userIDX(:,u));
   D = itemU(:,itemlist) + F2(:,itemlist);
   pre = D'*userU(:,u) + h(userIDX(:,u));
   [~,index] = sort(pre,'descend');
   gain = target(index);
   igain = sort(target,'descend');
   dcg = 0;
   idcg = 0;
   for k = 1:kvalue
       if k>length(gain)
           break;
       end
       dcg = dcg + (2.^gain(k)-1)/log2(1+k);
       idcg = idcg + (2.^igain(k)-1)/log2(1+k);
   end
   if idcg > 0
       ndcg(u) = dcg/idcg;
   else if idcg == 0
           invalid = invalid + 1;
        end
   end
   
end

ndcgvalue = sum(ndcg)/(unum - invalid);

end

