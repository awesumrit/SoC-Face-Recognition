function [person_no,face_no,minDistance] = classify(X,U,K,Z,gnd,X_norm)
 
%%%Normalise.............

  mu = mean(X);
  X_norm2 = bsxfun(@minus, X, mu);

  sigma = std(X_norm2);
  X_norm2 = bsxfun(@rdivide, X_norm2, sigma);
  
  
  
%%%Project on the face space....

  x=X_norm2 ;
  W = zeros(1,K);
  for j=1:K
    W(1,j) = x*U(:,j) ;  %  W is the projection (weight vector) for input 
                         %  test image
  endfor
  
%% Find the distance measure from other weight vectors of the dataset using
%% Euclidean distance and store the least distance measure
  
  m = size(Z,1) ;
  min = sqrt(sum((Z(1,:)-W(1,:)).^2))
  face = 1;
  for i=2:m
    min2 = sqrt(sum((Z(i,:)-W(1,:)).^2));
    if(min2<min)
      min = min2
      face = i;
    endif
  endfor
  
%%Recognize the image with least distance from the input test image

  face_no = face;
  person_no = gnd(face);
  minDistance = min;
  
  i= face;
  X_rec = ones(1,size(U,1));
  v = Z(i, :)' ;
  for j=1:size(U,1)
    X_rec(1,j) = v' * U(j, 1:K)' ;
  endfor
  
  subplot(1, 2, 1);
  displayData(X_norm(1:36,:));
  title('Original faces');
  axis square;
  
  subplot(1, 2, 2);
  displayData(X_rec);
  title('Recognized face');
  
endfunction
  
  
   