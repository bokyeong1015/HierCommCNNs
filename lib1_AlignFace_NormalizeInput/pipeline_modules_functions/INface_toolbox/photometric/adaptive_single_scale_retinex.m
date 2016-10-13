% The function applies the adaptive single scale retinex algorithm to an image.
% 
% PROTOTYPE
% [R,L] = adaptive_single_scale_retinex(X,T,S,h,normalize)
% 
% USAGE EXAMPLE(S)
% 
%     Example 1:
%       X=imread('sample_image.bmp');
%       Y = adaptive_single_scale_retinex(X);
%       figure,imshow(X);
%       figure,imshow(Y,[]);
% 
%     Example 2:
%       X=imread('sample_image.bmp');
%       Y = adaptive_single_scale_retinex(X,15);
%       figure,imshow(X);
%       figure,imshow(Y,[]);
%
%      Example 3:
%       X=imread('sample_image.bmp');
%       Y = adaptive_single_scale_retinex(X,10,10,1);
%       figure,imshow(X);
%       figure,imshow(Y,[]);
% 
%      Example 4:
%       X=imread('sample_image.bmp');
%       [R,L] = adaptive_single_scale_retinex(X);
%       figure,imshow(X);
%       figure,imshow(R,[]);
%       figure,imshow(L,[]);
% 
%      Example 5:
%       X=imread('sample_image.bmp');
%       [R,L] = adaptive_single_scale_retinex(X,[],[],[],0);
%       figure,imshow(X);
%       figure,imshow(R,[]);
%       figure,imshow(L,[]);
% 
%      Example 6:
%       X=imread('sample_image.bmp');
%       [R,L] = adaptive_single_scale_retinex(X,10,10,1,0);
%       figure,imshow(X);
%       figure,imshow(R,[]);
%       figure,imshow(L,[]);
%
%
% GENERAL DESCRIPTION
% The function performs photometric normalization of the image X using the
% ASR technique. It takes either one, two, four or five arguments with the first being
% the image to be normalized, the second being the number of iterations for
% the iterative convolution and the third and fourth being the parameters
% of the techqnique as defined in the original paper, where the
% normalization was proposed. The function cannot take three input
% arguments.
%
% The function is intended for use in face recognition experiments and the
% default parameters are set as proposed in the original paper. The names
% of the parameters (T, S, and h) were selcted in accrodance with the paper. 
%
%
%
% 
% REFERENCES
% This function is an implementation of the adaptive single scale retinex
% algorithm proposed in:
%
% Y.K. Park, S.L. Park, and J.K. Kim, �Retinex Method Based on Adaptive
% smoothing for Illumination Invariant Face Recognition,� Signal Processing, 
% vol. 88, no.8, pp. 1929-1945, 2008.
%
%
%
% INPUTS:
% X                     - a grey-scale image of arbitrary size
% T                     - an integer determining the number of iterative
%                         convolutions to perform (default: T = 10)
% S                     - an integer determining a weight needed for the
%                         filtering (default: S=10*exp(-(mean(I(:))/10)));
%                         here, the default value is set in the code and
%                         "I" is a variable set during computation
% h                     - an integer determining a weight needed for the
%                         filtering (default:
%                         h=0.1*exp(-(mean(tao_slash(:))/0.1))); here, the 
%                         default value is set in the code and "tao_slash" 
%                         is a variable set during computation
% normalize             - a parameter controlling the post-processing
%                         procedure:
%                            0 - no normalization
%                            1 - perform basic normalization (truncation 
%                                of histograms ends and normalization to the 
%                                8-bit interval) - default
%
%
% OUTPUTS:
% R                     - a grey-scale image processed with the ASR
%                         algorithm (the reflectance)
% L                     - the estimated luminance function
%
% NOTES / COMMENTS
% This function applies the adaptive single scale retinex algorithm on the
% grey-scale image X. 
%
% The function was tested with Matlab ver. 7.5.0.342 (R2007b) and Matlab 
% ver. 7.11.0.584 (R2010b).
%
% 
% RELATED FUNCTIONS (SEE ALSO)
% histtruncate  - a function provided by Peter Kovesi
% normalize8    - auxilary function
% 
% ABOUT
% Created:        20.8.2009
% Last Update:    26.1.2012
% Revision:       2.1
% 
%
% WHEN PUBLISHING A PAPER AS A RESULT OF RESEARCH CONDUCTED BY USING THIS CODE
% OR ANY PART OF IT, MAKE A REFERENCE TO THE FOLLOWING PUBLICATIONS:
%
% 1. �truc V., Pave�ic, N.:Photometric normalization techniques for illumination 
% invariance, in: Y.J. Zhang (Ed), Advances in Face Image Analysis: Techniques 
% and Technologies, IGI Global, pp. 279-300, 2011.
% (BibTex available from: http://luks.fe.uni-lj.si/sl/osebje/vitomir/pub/IGI.bib)
% 
% 2. �truc, V., Pave�ic, N.: Gabor-based kernel-partial-least-squares 
% discrimination features for face recognition, Informatica (Vilnius), 
% vol. 20, no. 1, pp. 115-138, 2009.
% (BibTex available from: http://luks.fe.uni-lj.si/sl/osebje/vitomir/pub/InforVI.bib)
% 
%
% Official website:
% If you have down-loaded the toolbox from any other location than the
% official website, plese check the following link to make sure that you
% have the most recent version:
% 
% http://luks.fe.uni-lj.si/sl/osebje/vitomir/face_tools/INFace/index.html
% 
% 
% Copyright (c) 2012 Vitomir �truc
% Faculty of Electrical Engineering,
% University of Ljubljana, Slovenia
% http://luks.fe.uni-lj.si/en/staff/vitomir/index.html
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files, to deal
% in the Software without restriction, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in 
% all copies or substantial portions of the Software.
%
% The Software is provided "as is", without warranty of any kind.
% 
% January 2012
function [R,L] = adaptive_single_scale_retinex(X,T,S,h,normalize)

%% Parameter checking
flag_p = 0;
if nargin==1
    flag_p = 1;
    normalize=1;
elseif nargin == 2
    flag_p = 2;
    normalize=1;
elseif nargin == 3
    disp('Not enough input parameters.')
    return;
elseif nargin == 4
    normalize = 1;
elseif nargin == 5    
    if isempty(T) && isempty(S) && isempty(h)
        flag_p = 1;
    elseif ~isempty(T) && ~isempty(S) && ~isempty(h)
        %proceed
    else
        disp('Error: when using this configuration of input parameters, T,S,h all need to be empty or none of them.')
        return;
    end
    if ~(normalize==1 || normalize==0)
        disp('Error: The fourth parameter can only be 0 or 1.');
        return;
     end
elseif nargin > 5
    disp('Error: To many input parameters.')
    return;
else
    flag_p = 0;
end

%% Init. operations
[a,b]=size(X);
X=double(normalize8(X));

%% Compute spatial gradient in x and y directions
X1=zeros(a,b+2);
X1(1:a,3:b+2)=X;

X2=zeros(a,b+2);
X2(1:a,1:b)=X;

Gx = X1(:,2:b+1)-X2(:,2:b+1);


X1=zeros(a+2,b);
X1(3:a+2,1:b)=X;

X2=zeros(a+2,b);
X2(1:a,1:b)=X;

Gy = X1(2:a+1,:)-X2(2:a+1,:);

I=sqrt(Gx.^2+Gy.^2);

%% Compute local inhomogenity
tao = zeros(a,b);
Xtmp=zeros(a+2,b+2);
Xtmp(2:a+1,2:b+1)=X;

for i=2:a+1
    for j=2:b+1
        suma=0;
        for k=-1:1:1
            for h=-1:1:1
                suma = suma+abs(X(i-1,j-1)-Xtmp(i+k,j+h));
            end
        end
        suma=suma/9;
        tao(i-1,j-1)=suma;
    end
end
tao_min=min(min(tao));
tao_max=max(max(tao));

tao_slash = (tao-tao_min)/(tao_max-tao_min);
tao_slash = sin(pi/2*tao_slash);


%% Set needed parameters if they are not provided as inputs
if flag_p == 1
    S=10*exp(-(mean(I(:))/10));
    h=0.1*exp(-(mean(tao_slash(:))/0.1));
    T=10;
elseif flag_p == 2
    S=10*exp(-(mean(I(:))/10));
    h=0.1*exp(-(mean(tao_slash(:))/0.1));
end


%% Determine weight functions
alpha = 1./(1+sqrt(tao_slash)/h);
beta = 1./(1+sqrt(I/S));
weight = alpha.*beta;

%precompute Ns
w=zeros(a+2,b+2);
w(2:a+1,2:b+1) = weight;
N=zeros(a,b);
for i=2:a+1
    for j=2:b+1
        suma = 0;
        for k=-1:1:1
            for h=-1:1:1
                suma = suma + (w(i+k,j+h));
            end
        end
        N(i-1,j-1)=suma;
    end
end


%% Start iterative convolution
L_old = X;
for i=1:T
    L_new_s = convolute(L_old,weight,N);
    L_new = reshape(max(L_new_s(:),L_old(:)),[a,b]);
    L_old=L_new; 
end

%% Produce ilumination invariant representation of input image
R = log(X+1)-log(L_new+1);
L = log(L_new+1);

%% Do some final post-procesing or not - you can comment this line out
if normalize ~= 0
    R=normalize8(R);
    [R, dummy] = histtruncate(R,0.2,0.2);
    R = normalize8(R);
    L = normalize8(L);
end






%% This is an auxialry function for computing the iterative convolution
function Y=convolute(X,y,N);

[a,b]=size(X);

X1=zeros(a+2,b+2);
X1(2:a+1,2:b+1) = X;

w=zeros(a+2,b+2);
w(2:a+1,2:b+1) = y;
Y=zeros(a,b);
for i=2:a+1
    for j=2:b+1
        Y(i-1,j-1)=(sum(sum(X1(i-1:i+1,j-1:j+1).*w(i-1:i+1,j-1:j+1))))/N(i-1,j-1);
    end
end
























