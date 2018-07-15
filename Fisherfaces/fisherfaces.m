clc;
clear;

load('Yale_32x32.mat');

Test_Set_Size = 200;

% Selecting the test set images
Test_Set_Indices = randperm(numel(gnd),Test_Set_Size);
Test_Set_Image = double(zeros(Test_Set_Size,numel(fea)/numel(gnd)));
Test_Set_Label = double(zeros(Test_Set_Size,1));

% Selecting the training set images
Training_Set_Image = double(zeros(numel(gnd)-Test_Set_Size,numel(fea)/numel(gnd)));
Training_Set_Label = double(zeros(numel(gnd)-Test_Set_Size,1));

Index = double(zeros(numel(gnd),1));

for i = 1:numel(gnd)
    Index(i) = i;
end

Bool_Training_Set = ismember(Index,Test_Set_Indices);

Count1 = 1;
Count2 = 1;

for i = 1:numel(Bool_Training_Set)
    if(Bool_Training_Set(i) == 1) 
       Test_Set_Image(Count1,:) = fea(i,:);
       Test_Set_Label(Count1) = gnd(i);
       Count1 = Count1 + 1;
    else
        Training_Set_Image(Count2,:) = fea(i,:);
        Training_Set_Label(Count2) = gnd(i);
        Count2 = Count2 + 1;
    end
end

Training_Set_Size = numel(Training_Set_Label);
Number_of_Classes = max(Training_Set_Label);

Image_Dimension = numel(Training_Set_Image)/Training_Set_Size;

% Classifying Images into Classes
Samples_Each_Class = double(zeros(Number_of_Classes,1));

for i = 1:Training_Set_Size
   Samples_Each_Class(Training_Set_Label(i)) = Samples_Each_Class(Training_Set_Label(i)) + 1;
end

for i = 1:Number_of_Classes
    Image_Class{i} = double(zeros(Samples_Each_Class(i),Image_Dimension));
end

Samples_Each_Class_z = Samples_Each_Class;

for i=1:Training_Set_Size
    label = Training_Set_Label(i);
    Image_Class{label}(Samples_Each_Class_z(label),:) = Training_Set_Image(i,:);
    Samples_Each_Class_z(label) = Samples_Each_Class_z(label) - 1;
end

% No need for PCA as Number of images greater than number of pixels

% Calculating Sb and Sw
Sb = double(zeros(Image_Dimension,Image_Dimension));
Sw = double(zeros(Image_Dimension,Image_Dimension));

Total_Mean = mean(Training_Set_Image);

for i=1:Number_of_Classes
    Mean_Vector = mean(Image_Class{i});
    Cov_Matrix = cov(Image_Class{i});
    Ni = Samples_Each_Class(i);
    Sb = Sb + (Ni*((Mean_Vector-Total_Mean)'*(Mean_Vector-Total_Mean)));
    Sw = Sw + (Ni-1)*Cov_Matrix;
end

% Calculating the generalised eigenvectors
[Gen_EigenVectors,Gen_EigenValues] = eig(Sb,Sw);

% Selecting the last c-1 Generalised Eigenvectors 
FLD_Matrix = Gen_EigenVectors(:,Image_Dimension-Number_of_Classes+2:Image_Dimension)';

% Generating the Normalised Dataset
Normalised_Training_Set = Training_Set_Image;
for i=1:Training_Set_Size
    Normalised_Training_Set(i,:) = Training_Set_Image(i,:) - Total_Mean;
end

% Taking Component of the Images along the FLD basis after normalisation 
Component_Data_Set = FLD_Matrix*Normalised_Training_Set';

% For each image of the test image, we subtract the total mean, then take
% the component along the FLD_Matrix and find the image amongst the component dataset
% which has minimum L2 norm with the test image

Predicted_Label = double(zeros(200,1));
Component_Test = double(zeros(Number_of_Classes-1,1));

for i = 1:Test_Set_Size
    Component_Test = FLD_Matrix*(Test_Set_Image(i,:)-Total_Mean)';
    Min_Norm = sqrt(sum((Component_Test - Component_Data_Set(:,1)).^2));
    Predicted_Label(i) = Training_Set_Label(1);
    for j = 1:Training_Set_Size
        Norm = sqrt(sum((Component_Test - Component_Data_Set(:,j)).^2));
        if(Min_Norm > Norm)
            Min_Norm = Norm;
            Predicted_Label(i) = Training_Set_Label(j); 
        end
    end
end

Count = 0;

for i=1:Test_Set_Size
    if(Test_Set_Label(i) == Predicted_Label(i))
        Count = Count + 1;
    end
end


